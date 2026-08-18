"""Train one MLX LoRA adapter on all assistant turns per trajectory."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import (
    linear_to_lora_layers,
    print_trainable_parameters,
)
from mlx_lm.utils import load

LORA_PARAMETERS = {"rank": 8, "dropout": 0.0, "scale": 20.0}


@dataclass(frozen=True)
class MaskedTrainingFailure:
    reason: str


@dataclass(frozen=True)
class MaskedSequence:
    tokens: tuple[int, ...]
    assistant_targets: tuple[bool, ...]


@dataclass(frozen=True)
class MaskedTrainingConfig:
    fine_tune_type: Literal["lora"]
    model: str
    num_layers: int
    lora_parameters: dict[str, float | int]
    objective: Literal["all_assistant_turns"]
    iterations: int
    learning_rate: float
    max_seq_length: int
    seed: int


@dataclass(frozen=True)
class Conversation:
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] | None


class ChatTokenizer(Protocol):
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None,
        add_generation_prompt: bool = False,
        return_dict: Literal[False] = False,
    ) -> list[int]: ...


class DistributedGroup(Protocol):
    def size(self) -> int: ...


class CausalModel(Protocol):
    def __call__(self, inputs: mx.array) -> mx.array: ...


class MaskedDataset:
    def __init__(
        self,
        sequences: Sequence[MaskedSequence],
        *,
        dropped_sequences: int,
    ) -> None:
        self._sequences = tuple(sequences)
        self.dropped_sequences = dropped_sequences

    def __getitem__(self, index: int) -> MaskedSequence:
        return self._sequences[index]

    def __len__(self) -> int:
        return len(self._sequences)


def _is_integer_token(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _token_list(
    value: object,
    *,
    label: str,
) -> list[int] | MaskedTrainingFailure:
    if not isinstance(value, list):
        return MaskedTrainingFailure(f"{label} tokens are invalid")
    if not all(_is_integer_token(token) for token in value):
        return MaskedTrainingFailure(f"{label} tokens are invalid")
    return value


def _conversation(
    row: Mapping[str, object],
) -> Conversation | MaskedTrainingFailure:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return MaskedTrainingFailure("trajectory messages are missing")
    if not all(isinstance(message, Mapping) for message in raw_messages):
        return MaskedTrainingFailure("trajectory message is invalid")
    messages = tuple(raw_messages)
    raw_tools = row.get("tools")
    if raw_tools is None:
        return Conversation(messages=messages, tools=None)
    if not isinstance(raw_tools, list) or not all(
        isinstance(tool, Mapping) for tool in raw_tools
    ):
        return MaskedTrainingFailure("trajectory tools are invalid")
    return Conversation(messages=messages, tools=tuple(raw_tools))


def _render_tokens(
    tokenizer: ChatTokenizer,
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[Mapping[str, object]] | None,
    *,
    label: str,
    add_generation_prompt: bool = False,
) -> list[int] | MaskedTrainingFailure:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
    except Exception as error:  # noqa: BLE001 - tokenizer boundary.
        return MaskedTrainingFailure(f"{label} tokenization failed: {error}")
    return _token_list(rendered, label=label)


def _assistant_span(
    tokenizer: ChatTokenizer,
    conversation: Conversation,
    full_tokens: list[int],
    message_index: int,
) -> tuple[int, int] | MaskedTrainingFailure:
    prompt_tokens = _render_tokens(
        tokenizer,
        conversation.messages[:message_index],
        conversation.tools,
        label="assistant prompt",
        add_generation_prompt=True,
    )
    if isinstance(prompt_tokens, MaskedTrainingFailure):
        return prompt_tokens
    start = len(prompt_tokens)
    if full_tokens[:start] != prompt_tokens:
        return MaskedTrainingFailure("assistant prompt is not a token prefix")
    eos_token_id = tokenizer.eos_token_id
    if not _is_integer_token(eos_token_id):
        return MaskedTrainingFailure("tokenizer end token is invalid")
    try:
        end = full_tokens.index(eos_token_id, start) + 1
    except ValueError:
        return MaskedTrainingFailure("assistant turn has no end token")
    if end <= start:
        return MaskedTrainingFailure("assistant turn is empty")
    return start, end


def _masked_sequence(
    tokenizer: ChatTokenizer,
    row: Mapping[str, object],
) -> MaskedSequence | MaskedTrainingFailure:
    conversation = _conversation(row)
    if isinstance(conversation, MaskedTrainingFailure):
        return conversation
    full_tokens = _render_tokens(
        tokenizer,
        conversation.messages,
        conversation.tools,
        label="trajectory",
    )
    if isinstance(full_tokens, MaskedTrainingFailure):
        return full_tokens
    assistant_targets = [False] * len(full_tokens)
    assistant_turns = 0
    for index, message in enumerate(conversation.messages):
        if message.get("role") != "assistant":
            continue
        span = _assistant_span(
            tokenizer,
            conversation,
            full_tokens,
            index,
        )
        if isinstance(span, MaskedTrainingFailure):
            return span
        for token_index in range(*span):
            assistant_targets[token_index] = True
        assistant_turns += 1
    if assistant_turns == 0 or not any(assistant_targets[1:]):
        return MaskedTrainingFailure("trajectory has no assistant targets")
    return MaskedSequence(
        tokens=tuple(full_tokens),
        assistant_targets=tuple(assistant_targets),
    )


def _row_failure(
    path: Path,
    line_number: int,
    detail: str,
) -> MaskedTrainingFailure:
    reason = f"training data row {line_number} in {path}: {detail}"
    return MaskedTrainingFailure(reason)


def _load_dataset(
    path: Path,
    tokenizer: ChatTokenizer,
    *,
    max_sequence_length: int,
) -> MaskedDataset | MaskedTrainingFailure:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        reason = f"training data is not readable: {path}: {error}"
        return MaskedTrainingFailure(reason)
    sequences: list[MaskedSequence] = []
    dropped_sequences = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            reason = f"invalid JSON at {path}:{line_number}: {error}"
            return MaskedTrainingFailure(reason)
        if not isinstance(row, Mapping):
            reason = f"invalid training data row at {path}:{line_number}"
            return MaskedTrainingFailure(reason)
        sequence = _masked_sequence(tokenizer, row)
        if isinstance(sequence, MaskedTrainingFailure):
            detail = f"conversion failed: {sequence.reason}"
            return _row_failure(path, line_number, detail)
        if len(sequence.tokens) > max_sequence_length:
            dropped_sequences += 1
            continue
        sequences.append(sequence)
    if not sequences:
        return MaskedTrainingFailure(f"no training sample fits the limit: {path}")
    return MaskedDataset(sequences, dropped_sequences=dropped_sequences)


def _iterate_masked_batches(
    dataset: MaskedDataset,
    batch_size: int,
    max_seq_length: int,
    *,
    loop: bool = False,
    comm_group: DistributedGroup | None = None,
) -> Iterator[tuple[mx.array, mx.array]]:
    if batch_size != 1:
        message = "assistant-masked training requires batch size 1"
        raise ValueError(message)
    if comm_group is not None and comm_group.size() != 1:
        message = "assistant-masked training does not support distribution"
        raise ValueError(message)
    generator = random.Random(0)  # noqa: S311 - deterministic training order.
    indices = list(range(len(dataset)))
    while True:
        generator.shuffle(indices)
        for index in indices:
            sequence = dataset[index]
            length = len(sequence.tokens)
            if length > max_seq_length:
                message = "masked training sequence exceeds its limit"
                raise ValueError(message)
            padded_length = min(
                1 + 32 * ((length + 31) // 32),
                max_seq_length,
            )
            padding = padded_length - length
            padded_tokens = list(sequence.tokens) + [0] * padding
            padded_targets = list(sequence.assistant_targets) + [False] * (
                padded_length - length
            )
            yield (
                mx.array([padded_tokens], dtype=mx.int32),
                mx.array([padded_targets], dtype=mx.bool_),
            )
        if not loop:
            break


def _assistant_loss(
    model: CausalModel,
    batch: mx.array,
    assistant_targets: mx.array,
) -> tuple[mx.array, mx.array]:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    target_mask = assistant_targets[:, 1:]
    logits = model(inputs)
    cross_entropy = nn.losses.cross_entropy(logits, targets) * target_mask
    token_count = target_mask.sum()
    loss = cross_entropy.astype(mx.float32).sum() / token_count
    return loss, token_count


def _run(arguments: argparse.Namespace) -> MaskedTrainingFailure | None:
    if arguments.iterations < 2:
        return MaskedTrainingFailure("iterations must be at least 2")
    if arguments.number_of_layers == 0 or arguments.number_of_layers < -1:
        return MaskedTrainingFailure("number of layers must be -1 or positive")
    finite_learning_rate = math.isfinite(arguments.learning_rate)
    positive_learning_rate = arguments.learning_rate > 0
    if not finite_learning_rate or not positive_learning_rate:
        reason = "learning rate must be positive and finite"
        return MaskedTrainingFailure(reason)
    if arguments.max_sequence_length < 2:
        return MaskedTrainingFailure("sequence length must be at least 2")
    if arguments.checkpoint_interval < 1:
        return MaskedTrainingFailure("checkpoint interval must be positive")

    mx.random.seed(arguments.seed)
    print("Loading pretrained model", flush=True)
    model, tokenizer = load(
        arguments.model,
        tokenizer_config={"trust_remote_code": True},
    )
    if arguments.number_of_layers > len(model.layers):
        reason = "number of layers exceeds the model depth"
        return MaskedTrainingFailure(reason)
    train_dataset = _load_dataset(
        arguments.data / "train.jsonl",
        tokenizer,
        max_sequence_length=arguments.max_sequence_length,
    )
    if isinstance(train_dataset, MaskedTrainingFailure):
        return train_dataset
    validation_dataset = _load_dataset(
        arguments.data / "valid.jsonl",
        tokenizer,
        max_sequence_length=arguments.max_sequence_length,
    )
    if isinstance(validation_dataset, MaskedTrainingFailure):
        return validation_dataset
    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "train_samples_dropped": train_dataset.dropped_sequences,
                "validation_samples": len(validation_dataset),
                "validation_samples_dropped": validation_dataset.dropped_sequences,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model.freeze()
    linear_to_lora_layers(
        model,
        arguments.number_of_layers,
        LORA_PARAMETERS,
        use_dora=False,
    )
    print_trainable_parameters(model)
    arguments.adapter_path.mkdir(parents=True)
    config = MaskedTrainingConfig(
        fine_tune_type="lora",
        model=arguments.model,
        num_layers=arguments.number_of_layers,
        lora_parameters=LORA_PARAMETERS,
        objective="all_assistant_turns",
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        max_seq_length=arguments.max_sequence_length,
        seed=arguments.seed,
    )
    (arguments.adapter_path / "adapter_config.json").write_text(
        json.dumps(
            asdict(config),
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    training_arguments = TrainingArgs(
        batch_size=1,
        iters=arguments.iterations,
        val_batches=1,
        steps_per_report=1,
        steps_per_eval=arguments.checkpoint_interval,
        steps_per_save=arguments.checkpoint_interval,
        adapter_file=arguments.adapter_path / "adapters.safetensors",
        max_seq_length=arguments.max_sequence_length,
        grad_checkpoint=True,
    )
    optimizer = optim.Adam(learning_rate=arguments.learning_rate)
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        args=training_arguments,
        loss=_assistant_loss,
        iterate_batches=_iterate_masked_batches,
    )
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-masked-lora")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--num-layers", dest="number_of_layers", type=int, required=True
    )
    parser.add_argument("--iters", dest="iterations", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max-seq-length",
        dest="max_sequence_length",
        type=int,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parser().parse_args(argv))
    if result is None:
        return 0
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
