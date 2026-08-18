import assert from "node:assert/strict";

import { parseQuery } from "./query_string.mjs";

const CASES = [
  ["", {}],
  ["?", {}],
  ["a=1", { a: "1" }],
  ["?a=1&b=2", { a: "1", b: "2" }],
  ["flag", { flag: "" }],
  ["a=1&a=2", { a: "2" }],
  ["a%20b=c%26d", { "a b": "c&d" }],
  ["&&a=1&&", { a: "1" }],
  ["a=b=c", { a: "b=c" }],
  ["?x=&y", { x: "", y: "" }],
];

for (const [query, expected] of CASES) {
  const label = `parseQuery(${JSON.stringify(query)})`;
  assert.deepEqual(parseQuery(query), expected, label);
}
console.log(`all ${CASES.length} cases passed`);
