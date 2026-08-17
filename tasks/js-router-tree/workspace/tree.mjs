// The route tree and its registration rules.
//
// A node is { children: Map, param: null | { name, node },
// wildcard: null | { name, handler }, handler: null | handler }.
// children maps a static segment string to a child node.
//
// - createNode() returns one fresh empty node.
// - insertRoute(root, segments, handler) walks the segments and
//   builds nodes on the way:
//   * A segment ":name" uses the param slot. One node has at most
//     one param name: a second registration through the same node
//     with a different name throws
//     Error("conflicting param names").
//   * A segment "*name" uses the wildcard slot and must be the last
//     segment; otherwise it throws
//     Error("wildcard must be the last segment"). A second wildcard
//     on the same node throws Error("duplicate route").
//   * Any other segment is static.
//   * When the final node already has a handler, insertRoute throws
//     Error("duplicate route").

export function createNode() {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}

export function insertRoute(root, segments, handler) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}
