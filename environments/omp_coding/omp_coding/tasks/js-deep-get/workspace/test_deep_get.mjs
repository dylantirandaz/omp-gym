import assert from "node:assert/strict";

import { deepGet } from "./deep_get.mjs";

const data = {
  a: { b: [{ c: 42 }, { c: 7 }] },
  count: 0,
  name: "",
  empty: null,
};

assert.equal(deepGet(data, "a.b.0.c", "missing"), 42, "nested array step");
assert.equal(deepGet(data, "a.b.1.c", "missing"), 7, "second array index");
assert.deepEqual(deepGet(data, "a.b.0", "missing"), { c: 42 }, "object result");
assert.equal(deepGet(data, "count", "missing"), 0, "falsy number is found");
assert.equal(deepGet(data, "name", "missing"), "", "falsy string is found");
assert.equal(deepGet(data, "a.x", "missing"), "missing", "missing key");
assert.equal(deepGet(data, "a.b.2.c", "missing"), "missing", "index out of range");
assert.equal(deepGet(data, "empty", "missing"), "missing", "null value");
assert.equal(deepGet(data, "empty.deep", "missing"), "missing", "step through null");
assert.equal(deepGet(data, "count.digits", "missing"), "missing", "step through number");
assert.equal(deepGet(null, "a", "missing"), "missing", "null target");
console.log("all 11 cases passed");
