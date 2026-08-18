// Execute black-box Node calls inside a candidate container.

import { open } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const MAX_DEPTH = 32;
const MAX_ITEMS = 10000;
const MAX_REQUEST_BYTES = 2 * 1024 * 1024;
const WORKSPACE = "/workspace";

function abort(reason) {
  process.stdout.write(
    `${JSON.stringify({ schema_version: 1, status: "worker_error", reason })}\n`,
  );
  process.exit(2);
}

const moduleCache = new Map();

async function target(value) {
  if (typeof value !== "string" || value.split(":").length !== 2) {
    abort("target must use file:name syntax");
  }
  const [fileName, qualifiedName] = value.split(":", 2);
  const parts = fileName.split("/");
  if (
    !fileName.startsWith("./") ||
    !fileName.endsWith(".mjs") ||
    parts.includes("..") ||
    qualifiedName.length === 0
  ) {
    abort("target file or name is invalid");
  }
  let moduleValue = moduleCache.get(fileName);
  if (moduleValue === undefined) {
    moduleValue = await import(pathToFileURL(`${WORKSPACE}/${fileName.slice(2)}`).href);
    moduleCache.set(fileName, moduleValue);
  }
  let current = moduleValue;
  for (const part of qualifiedName.split(".")) {
    if (part.length === 0 || part.startsWith("_")) {
      abort("target contains a private or empty name");
    }
    current = current[part];
  }
  return current;
}

function decode(value, values, callbackArguments = [], depth = 0) {
  if (depth > MAX_DEPTH || value === null || typeof value !== "object") {
    abort("encoded value is invalid");
  }
  const kind = value.kind;
  if (kind === "none") return null;
  if (kind === "undefined") return undefined;
  if (kind === "bool" && typeof value.value === "boolean") return value.value;
  if (kind === "number" && typeof value.value === "string") {
    const result = Number(value.value);
    if (!Number.isFinite(result)) abort("encoded number is invalid");
    return result;
  }
  if (kind === "bigint" && typeof value.value === "string") {
    try {
      return BigInt(value.value);
    } catch {
      abort("encoded bigint is invalid");
    }
  }
  if (kind === "str" && typeof value.value === "string") return value.value;
  if (kind === "ref" && typeof value.name === "string") {
    if (!values.has(value.name)) abort(`unknown value reference: ${value.name}`);
    return values.get(value.name);
  }
  if (kind === "callback_arg" && Number.isSafeInteger(value.index)) {
    if (value.index < 0 || value.index >= callbackArguments.length) {
      abort("callback argument index is invalid");
    }
    return callbackArguments[value.index];
  }
  if (kind === "callback_args") return [...callbackArguments];
  if (kind === "list") {
    if (!Array.isArray(value.items) || value.items.length > MAX_ITEMS) {
      abort("encoded list is invalid");
    }
    return value.items.map((item) => decode(item, values, callbackArguments, depth + 1));
  }
  if (kind === "object") {
    if (value.fields === null || typeof value.fields !== "object" || Array.isArray(value.fields)) {
      abort("encoded object is invalid");
    }
    const output = {};
    for (const [name, item] of Object.entries(value.fields)) {
      output[name] = decode(item, values, callbackArguments, depth + 1);
    }
    return output;
  }
  if (kind === "map") {
    if (!Array.isArray(value.items) || value.items.length > MAX_ITEMS) {
      abort("encoded map is invalid");
    }
    return new Map(
      value.items.map((pair) => {
        if (!Array.isArray(pair) || pair.length !== 2) abort("encoded map item is invalid");
        return [
          decode(pair[0], values, callbackArguments, depth + 1),
          decode(pair[1], values, callbackArguments, depth + 1),
        ];
      }),
    );
  }
  abort("encoded value kind is invalid");
}

function encode(value, depth = 0, seen = new Set()) {
  if (depth > MAX_DEPTH) abort("candidate result is too deep");
  if (value === null) return { kind: "none" };
  if (value === undefined) return { kind: "undefined" };
  if (typeof value === "boolean") return { kind: "bool", value };
  if (typeof value === "number") {
    if (!Number.isFinite(value)) abort("candidate returned a nonfinite number");
    return { kind: "number", value: Object.is(value, -0) ? "-0" : String(value) };
  }
  if (typeof value === "bigint") return { kind: "bigint", value: String(value) };
  if (typeof value === "string") return { kind: "str", value };
  if (typeof value === "function") abort("candidate returned an observable function");
  if (typeof value !== "object") abort(`unsupported candidate type: ${typeof value}`);
  if (seen.has(value)) abort("candidate returned a cyclic value");
  seen.add(value);
  try {
    if (Array.isArray(value)) {
      if (value.length > MAX_ITEMS) abort("candidate returned too many list items");
      return { kind: "list", items: value.map((item) => encode(item, depth + 1, seen)) };
    }
    if (value instanceof Map) {
      if (value.size > MAX_ITEMS) abort("candidate returned too many map items");
      return {
        kind: "map",
        items: [...value.entries()].map(([key, item]) => [
          encode(key, depth + 1, seen),
          encode(item, depth + 1, seen),
        ]),
      };
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      abort(`candidate returned an unsupported object: ${value.constructor?.name ?? "unknown"}`);
    }
    const fields = {};
    for (const name of Object.keys(value).sort()) {
      fields[name] = encode(value[name], depth + 1, seen);
    }
    return { kind: "object", fields };
  } finally {
    seen.delete(value);
  }
}

function argumentsFor(operation, values, callbackArguments = []) {
  const rawArguments = operation.args ?? [];
  if (!Array.isArray(rawArguments)) abort("operation arguments are invalid");
  return rawArguments.map((item) => decode(item, values, callbackArguments));
}

function storeName(operation) {
  if (
    typeof operation.store !== "string" ||
    operation.store.length === 0 ||
    operation.store.startsWith("_")
  ) {
    abort("operation store name is invalid");
  }
  return operation.store;
}

function callbackActions(actions, values, callbackArguments) {
  if (!Array.isArray(actions) || actions.length > 100) abort("callback actions are invalid");
  for (const action of actions) {
    if (action === null || typeof action !== "object") abort("callback action is invalid");
    if (action.op === "record") {
      if (typeof action.list !== "string" || !values.has(action.list)) {
        abort("callback list reference is invalid");
      }
      const list = values.get(action.list);
      if (!Array.isArray(list)) abort("callback record target is not a list");
      list.push(decode(action.value, values, callbackArguments));
    } else if (action.op === "increment") {
      if (typeof action.value !== "string" || !values.has(action.value)) {
        abort("callback number reference is invalid");
      }
      const current = values.get(action.value);
      if (typeof current !== "number") abort("callback increment target is not a number");
      values.set(action.value, current + 1);
    } else if (action.op === "method") {
      const objectValue = decode(action.target, values, callbackArguments);
      if (typeof action.name !== "string" || action.name.startsWith("_")) {
        abort("callback method name is invalid");
      }
      const method = objectValue[action.name];
      if (typeof method !== "function") abort("callback method is not callable");
      const result = method.apply(
        objectValue,
        argumentsFor(action, values, callbackArguments),
      );
      if (typeof action.store === "string") values.set(action.store, result);
    } else if (action.op === "if_list_length") {
      if (
        typeof action.list !== "string" ||
        !values.has(action.list) ||
        !Number.isSafeInteger(action.equals)
      ) {
        abort("callback condition is invalid");
      }
      const list = values.get(action.list);
      if (!Array.isArray(list)) abort("callback condition target is not a list");
      if (list.length === action.equals) {
        callbackActions(action.then, values, callbackArguments);
      }
    } else {
      abort("callback action kind is invalid");
    }
  }
}

async function runOperation(operation, values) {
  if (operation === null || typeof operation !== "object") abort("operation is invalid");
  if (operation.op === "value") {
    values.set(storeName(operation), decode(operation.value, values));
    return;
  }
  if (operation.op === "callback") {
    const actions = operation.actions;
    values.set(storeName(operation), (...callbackArguments) => {
      callbackActions(actions, values, callbackArguments);
    });
    return;
  }
  if (operation.op === "call" || operation.op === "construct") {
    const callableValue = await target(operation.target);
    if (typeof callableValue !== "function") abort("operation target is not callable");
    const args = argumentsFor(operation, values);
    const result =
      operation.op === "construct" ? new callableValue(...args) : callableValue(...args);
    values.set(storeName(operation), await result);
    return;
  }
  if (operation.op === "method") {
    const objectValue = decode(operation.target, values);
    if (typeof operation.name !== "string" || operation.name.startsWith("_")) {
      abort("method name is invalid");
    }
    const method = objectValue[operation.name];
    if (typeof method !== "function") abort("method is not callable");
    values.set(
      storeName(operation),
      await method.apply(objectValue, argumentsFor(operation, values)),
    );
    return;
  }
  if (operation.op === "invoke") {
    const callableValue = decode(operation.target, values);
    if (typeof callableValue !== "function") abort("stored value is not callable");
    values.set(storeName(operation), await callableValue(...argumentsFor(operation, values)));
    return;
  }
  if (operation.op === "get") {
    const objectValue = decode(operation.target, values);
    if (typeof operation.property !== "string" || operation.property.startsWith("_")) {
      abort("property name is invalid");
    }
    values.set(storeName(operation), objectValue[operation.property]);
    return;
  }
  abort("operation kind is invalid");
}

async function main() {
  let requestBytes;
  if (process.argv.length === 2) {
    const chunks = [];
    let byteCount = 0;
    for await (const chunk of process.stdin) {
      byteCount += chunk.length;
      if (byteCount > MAX_REQUEST_BYTES) abort("request is too large");
      chunks.push(chunk);
    }
    requestBytes = Buffer.concat(chunks);
  } else if (process.argv.length === 3) {
    try {
      const handle = await open(process.argv[2], "r");
      try {
        const buffer = Buffer.alloc(MAX_REQUEST_BYTES + 1);
        const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
        requestBytes = buffer.subarray(0, bytesRead);
      } finally {
        await handle.close();
      }
    } catch (error) {
      abort(`request is not readable: ${error.message}`);
    }
  } else {
    abort("worker accepts zero or one request path");
  }
  if (requestBytes.length > MAX_REQUEST_BYTES) abort("request is too large");
  let request;
  try {
    request = JSON.parse(requestBytes.toString("utf8"));
  } catch (error) {
    abort(`request is invalid: ${error.message}`);
  }
  if (request === null || typeof request !== "object" || request.schema_version !== 1) {
    abort("request schema version is invalid");
  }
  if (
    !Array.isArray(request.operations) ||
    request.operations.length > 200 ||
    !Array.isArray(request.observe) ||
    !request.observe.every((name) => typeof name === "string")
  ) {
    abort("request operations or observations are invalid");
  }
  const values = new Map();
  for (let index = 0; index < request.operations.length; index += 1) {
    try {
      await runOperation(request.operations[index], values);
    } catch (error) {
      process.stdout.write(
        `${JSON.stringify({
          schema_version: 1,
          status: "error",
          operation: index,
          error: {
            type: error?.constructor?.name ?? "Error",
            message: String(error?.message ?? error),
          },
        })}\n`,
      );
      return;
    }
  }
  const output = {};
  for (const name of request.observe) {
    if (!values.has(name)) abort(`unknown observation: ${name}`);
    output[name] = encode(values.get(name));
  }
  process.stdout.write(
    `${JSON.stringify({ schema_version: 1, status: "ok", values: output })}\n`,
  );
}

await main();
