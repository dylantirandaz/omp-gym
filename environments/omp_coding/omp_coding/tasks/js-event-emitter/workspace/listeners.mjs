// Listener registry helpers for the emitter.
//
// The registry is a Map from event name to an array of entry objects
// { fn, once, fired }. The rules are:
//
// - addListener(registry, event, fn, once) appends one new entry
//   { fn, once, fired: false } and returns that entry.
// - removeListener(registry, event, entry) removes that exact entry
//   object (identity, not fn equality) and keeps the order of the
//   rest. A missing entry is a no-op.
// - removeFirstByFn(registry, event, fn) removes the first entry
//   whose fn is the given function. A missing fn is a no-op.
// - snapshotListeners(registry, event) returns a new array with the
//   current entries. Later registry changes never change the
//   snapshot. An unknown event gives [].

export function addListener(registry, event, fn, once) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}

export function removeListener(registry, event, entry) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}

export function removeFirstByFn(registry, event, fn) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}

export function snapshotListeners(registry, event) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}
