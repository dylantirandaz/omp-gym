// A small event emitter with strict reentrancy rules.
//
// - on(event, fn) registers fn and returns an unsubscribe function.
//   The unsubscribe function removes exactly the entry it created,
//   even when the same fn is registered several times.
// - once(event, fn) is like on, but the entry runs at most one time.
// - off(event, fn) removes the first entry for that event whose
//   handler is fn.
// - emit(event, ...args) takes a snapshot of the entries first, then
//   runs the snapshot in registration order and returns the number
//   of entries it ran. The rules during a running emit:
//   * An entry added during the emit does not run in this emit.
//   * An entry removed during the emit still runs: the snapshot won.
//   * A once entry leaves the registry before its handler runs, so
//     a reentrant emit inside the handler does not run it again.
//   * A once entry runs at most one time in total. When a nested
//     emit already ran it, the outer emit skips it and does not
//     count it.

import {
  addListener,
  removeFirstByFn,
  removeListener,
  snapshotListeners,
} from "./listeners.mjs";

export class Emitter {
  constructor() {
    this.registry = new Map();
  }

  on(event, fn) {
    // Not implemented yet. This is the task.
    throw new Error("Not implemented");
  }

  once(event, fn) {
    // Not implemented yet. This is the task.
    throw new Error("Not implemented");
  }

  off(event, fn) {
    // Not implemented yet. This is the task.
    throw new Error("Not implemented");
  }

  emit(event, ...args) {
    // Not implemented yet. This is the task.
    throw new Error("Not implemented");
  }
}
