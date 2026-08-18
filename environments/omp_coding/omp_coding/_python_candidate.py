"""Execute black-box Python calls inside a candidate container."""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

MAX_DEPTH = 32
MAX_ITEMS = 10_000
MAX_REQUEST_BYTES = 2 * 1024 * 1024

sys.path.insert(0, "/workspace")


def _abort(reason: str) -> NoReturn:
    print(
        json.dumps(
            {"schema_version": 1, "status": "worker_error", "reason": reason},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    raise SystemExit(2)


def _target(value: object) -> object:
    if not isinstance(value, str) or value.count(":") != 1:
        _abort("target must use module:name syntax")
    module_name, qualified_name = value.split(":", 1)
    if not module_name or not qualified_name:
        _abort("target must use module:name syntax")
    current: object = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        if not part or part.startswith("_"):
            _abort("target contains a private or empty name")
        current = getattr(current, part)
    return current


def _decode(value: object, values: Mapping[str, object], depth: int = 0) -> object:
    if depth > MAX_DEPTH or not isinstance(value, dict):
        _abort("encoded value is invalid")
    kind = value.get("kind")
    if kind == "none":
        return None
    if kind == "bool" and isinstance(value.get("value"), bool):
        return value["value"]
    if kind == "int" and isinstance(value.get("value"), str):
        try:
            return int(value["value"])
        except ValueError:
            _abort("encoded integer is invalid")
    if kind == "float" and isinstance(value.get("value"), str):
        try:
            result = float(value["value"])
        except ValueError:
            _abort("encoded float is invalid")
        if not math.isfinite(result):
            _abort("encoded float must be finite")
        return result
    if kind == "str" and isinstance(value.get("value"), str):
        return value["value"]
    if kind == "ref" and isinstance(value.get("name"), str):
        name = value["name"]
        if name not in values:
            _abort(f"unknown value reference: {name}")
        return values[name]
    if kind in {"list", "tuple"}:
        items = value.get("items")
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            _abort("encoded sequence is invalid")
        decoded = [_decode(item, values, depth + 1) for item in items]
        return decoded if kind == "list" else tuple(decoded)
    if kind == "dict":
        items = value.get("items")
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            _abort("encoded dictionary is invalid")
        output: dict[object, object] = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                _abort("encoded dictionary item is invalid")
            key = _decode(pair[0], values, depth + 1)
            output[key] = _decode(pair[1], values, depth + 1)
        return output
    if kind == "object":
        target = _target(value.get("type"))
        fields = value.get("fields")
        if not isinstance(fields, dict) or not all(
            isinstance(name, str) for name in fields
        ):
            _abort("encoded object fields are invalid")
        decoded_fields = {
            name: _decode(field, values, depth + 1) for name, field in fields.items()
        }
        if not callable(target):
            _abort("encoded object type is not callable")
        return target(**decoded_fields)
    _abort("encoded value kind is invalid")


def _encode(value: object, depth: int = 0) -> dict[str, object]:
    if depth > MAX_DEPTH:
        _abort("candidate result is too deep")
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            _abort("candidate returned a nonfinite float")
        return {"kind": "float", "value": repr(value)}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, list):
        if len(value) > MAX_ITEMS:
            _abort("candidate returned too many list items")
        return {"kind": "list", "items": [_encode(item, depth + 1) for item in value]}
    if isinstance(value, tuple):
        if len(value) > MAX_ITEMS:
            _abort("candidate returned too many tuple items")
        return {"kind": "tuple", "items": [_encode(item, depth + 1) for item in value]}
    if isinstance(value, dict):
        if len(value) > MAX_ITEMS:
            _abort("candidate returned too many dictionary items")
        pairs = [
            [_encode(key, depth + 1), _encode(item, depth + 1)]
            for key, item in value.items()
        ]
        pairs.sort(
            key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":"))
        )
        return {"kind": "dict", "items": pairs}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value_type = type(value)
        fields = {
            field.name: _encode(getattr(value, field.name), depth + 1)
            for field in dataclasses.fields(value)
        }
        return {
            "kind": "object",
            "type": f"{value_type.__module__}:{value_type.__qualname__}",
            "fields": fields,
        }
    _abort(f"candidate returned an unsupported type: {type(value).__name__}")


def _arguments(
    operation: Mapping[str, object], values: Mapping[str, object]
) -> tuple[list[object], dict[str, object]]:
    raw_arguments = operation.get("args", [])
    raw_keywords = operation.get("kwargs", {})
    if not isinstance(raw_arguments, list) or not isinstance(raw_keywords, dict):
        _abort("operation arguments are invalid")
    if not all(isinstance(name, str) for name in raw_keywords):
        _abort("operation keyword name is invalid")
    arguments = [_decode(item, values) for item in raw_arguments]
    keywords = {name: _decode(item, values) for name, item in raw_keywords.items()}
    return arguments, keywords


def _store_name(operation: Mapping[str, object]) -> str:
    name = operation.get("store")
    if not isinstance(name, str) or not name or name.startswith("_"):
        _abort("operation store name is invalid")
    return name


def _run_operation(operation: object, values: dict[str, object]) -> None:
    if not isinstance(operation, dict):
        _abort("operation must be an object")
    kind = operation.get("op")
    if kind == "value":
        values[_store_name(operation)] = _decode(operation.get("value"), values)
        return
    if kind in {"call", "construct"}:
        callable_value = _target(operation.get("target"))
        if not callable(callable_value):
            _abort("operation target is not callable")
        arguments, keywords = _arguments(operation, values)
        values[_store_name(operation)] = callable_value(*arguments, **keywords)
        return
    if kind == "method":
        object_value = _decode(operation.get("target"), values)
        method_name = operation.get("name")
        if (
            not isinstance(method_name, str)
            or not method_name
            or method_name.startswith("_")
        ):
            _abort("method name is invalid")
        method = getattr(object_value, method_name)
        if not callable(method):
            _abort("method is not callable")
        arguments, keywords = _arguments(operation, values)
        values[_store_name(operation)] = method(*arguments, **keywords)
        return
    if kind == "invoke":
        callable_value = _decode(operation.get("target"), values)
        if not callable(callable_value):
            _abort("stored value is not callable")
        arguments, keywords = _arguments(operation, values)
        values[_store_name(operation)] = callable_value(*arguments, **keywords)
        return
    if kind == "get":
        object_value = _decode(operation.get("target"), values)
        attribute = operation.get("property")
        if not isinstance(attribute, str) or not attribute or attribute.startswith("_"):
            _abort("attribute name is invalid")
        values[_store_name(operation)] = getattr(object_value, attribute)
        return
    _abort("operation kind is invalid")


def _request_bytes() -> bytes:
    if len(sys.argv) == 1:
        return sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(sys.argv) != 2:
        _abort("worker accepts zero or one request path")
    try:
        with Path(sys.argv[1]).open("rb") as stream:
            return stream.read(MAX_REQUEST_BYTES + 1)
    except OSError as error:
        _abort(f"request is not readable: {error}")


def main() -> int:
    raw = _request_bytes()
    if len(raw) > MAX_REQUEST_BYTES:
        _abort("request is too large")
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _abort(f"request is invalid: {error}")
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        _abort("request schema version is invalid")
    operations = request.get("operations")
    observe = request.get("observe")
    if (
        not isinstance(operations, list)
        or len(operations) > 200
        or not isinstance(observe, list)
        or not all(isinstance(name, str) for name in observe)
    ):
        _abort("request operations or observations are invalid")
    values: dict[str, object] = {}
    for index, operation in enumerate(operations):
        try:
            _run_operation(operation, values)
        except BaseException as error:
            error_type = type(error)
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "error",
                        "operation": index,
                        "error": {
                            "type": f"{error_type.__module__}:{error_type.__qualname__}",
                            "message": str(error),
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            return 0
    output: dict[str, object] = {}
    for name in observe:
        if name not in values:
            _abort(f"unknown observation: {name}")
        output[name] = _encode(values[name])
    print(
        json.dumps(
            {"schema_version": 1, "status": "ok", "values": output},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
