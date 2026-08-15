import assert from "node:assert/strict";

import { matchRoute } from "./route_match.mjs";

const cases = [
  ["root route", "/", "/", { matched: true, params: {} }],
  [
    "literal route",
    "/health/live",
    "/health/live",
    { matched: true, params: {} },
  ],
  ["literal mismatch", "/health/live", "/health/ready", { matched: false }],
  ["too few segments", "/users/:id", "/users", { matched: false }],
  ["too many segments", "/users/:id", "/users/7/edit", { matched: false }],
  [
    "one parameter",
    "/users/:id",
    "/users/42",
    { matched: true, params: { id: "42" } },
  ],
  [
    "two parameters",
    "/teams/:team/users/:user",
    "/teams/red/users/ava",
    { matched: true, params: { team: "red", user: "ava" } },
  ],
  [
    "encoded parameter",
    "/files/:name",
    "/files/quarter%20one.txt",
    { matched: true, params: { name: "quarter one.txt" } },
  ],
  [
    "trailing slash",
    "/users/:id",
    "/users/42/",
    { matched: true, params: { id: "42" } },
  ],
  [
    "empty inner segment",
    "/users/:id",
    "/users//42",
    { matched: false },
  ],
];

for (const [name, pattern, path, expected] of cases) {
  assert.deepEqual(matchRoute(pattern, path), expected, name);
}

console.log(`all ${cases.length} cases passed`);
