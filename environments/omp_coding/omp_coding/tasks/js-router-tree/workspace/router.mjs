// The public router: registration plus lookup with precedence.
//
// - splitPath(path) turns "/a/b" into ["a", "b"]. Empty segments
//   drop out, so "/", "/a/" and "/a" behave as expected.
// - createRouter() returns { register, resolve }.
//   * register(path, handler) inserts the route into one shared
//     tree with tree.insertRoute.
//   * resolve(path) returns { handler, params } or null. params maps
//     each ":name" to its segment and each "*name" to the joined
//     rest ("b/c" for segments b, c).
// - Precedence at every node: static beats param, param beats
//   wildcard. The search backtracks: when the static branch dead
//   ends deeper in the path, the param branch of the same node is
//   tried next, then the wildcard.
// - A wildcard needs at least one remaining segment: a route
//   "/files/*p" does not match "/files".
// - resolve never leaks params between calls.

import { createNode, insertRoute } from "./tree.mjs";

export function splitPath(path) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}

export function createRouter() {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}
