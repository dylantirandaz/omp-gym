import assert from "node:assert/strict";

import { Emitter } from "./emitter.mjs";
import { addListener, snapshotListeners } from "./listeners.mjs";

// registry helpers
{
  const registry = new Map();
  const entry = addListener(registry, "e", () => {}, false);
  assert.equal(entry.once, false, "entry records once flag");
  assert.equal(entry.fired, false, "entry starts unfired");
  const snap = snapshotListeners(registry, "e");
  addListener(registry, "e", () => {}, true);
  assert.equal(snap.length, 1, "snapshot ignores later adds");
  assert.deepEqual(snapshotListeners(registry, "nope"), [], "unknown event");
}

// basic emit order and count
{
  const emitter = new Emitter();
  const calls = [];
  emitter.on("tick", (v) => calls.push(["a", v]));
  emitter.on("tick", (v) => calls.push(["b", v]));
  assert.equal(emitter.emit("tick", 1), 2, "emit counts handlers");
  assert.deepEqual(calls, [["a", 1], ["b", 1]], "registration order");
  assert.equal(emitter.emit("other"), 0, "unknown event runs nothing");
}

// unsubscribe removes exactly its own entry
{
  const emitter = new Emitter();
  const calls = [];
  const fn = () => calls.push("x");
  emitter.on("e", fn);
  const unsubscribe = emitter.on("e", fn);
  unsubscribe();
  assert.equal(emitter.emit("e"), 1, "only the second entry left");
  unsubscribe();
  assert.equal(emitter.emit("e"), 1, "second unsubscribe is a no-op");
  assert.deepEqual(calls, ["x", "x"], "same fn stayed registered once");
}

// off removes the first matching entry
{
  const emitter = new Emitter();
  const calls = [];
  emitter.on("e", () => calls.push("first"));
  emitter.on("e", () => calls.push("second"));
  const shared = () => calls.push("shared");
  emitter.on("e", shared);
  emitter.on("e", shared);
  emitter.off("e", shared);
  assert.equal(emitter.emit("e"), 3, "one shared entry removed");
  assert.deepEqual(calls, ["first", "second", "shared"], "first match went");
}

// once runs one time
{
  const emitter = new Emitter();
  let count = 0;
  emitter.once("e", () => (count += 1));
  assert.equal(emitter.emit("e"), 1, "once ran");
  assert.equal(emitter.emit("e"), 0, "once is gone");
  assert.equal(count, 1, "once ran exactly one time");
}

// adds during emit wait for the next emit
{
  const emitter = new Emitter();
  const calls = [];
  emitter.on("e", () => {
    calls.push("outer");
    emitter.on("e", () => calls.push("inner"));
  });
  assert.equal(emitter.emit("e"), 1, "new entry did not run");
  assert.equal(emitter.emit("e"), 2, "new entry runs next time");
  assert.deepEqual(calls, ["outer", "outer", "inner"], "order kept");
}

// removals during emit still run from the snapshot
{
  const emitter = new Emitter();
  const calls = [];
  const second = () => calls.push("second");
  emitter.on("e", () => {
    calls.push("first");
    emitter.off("e", second);
  });
  emitter.on("e", second);
  assert.equal(emitter.emit("e"), 2, "snapshot kept the removed entry");
  assert.deepEqual(calls, ["first", "second"], "removed entry still ran");
  assert.equal(emitter.emit("e"), 1, "removal holds afterwards");
}

// reentrant emit inside a once handler does not loop
{
  const emitter = new Emitter();
  let count = 0;
  emitter.once("e", () => {
    count += 1;
    emitter.emit("e");
  });
  assert.equal(emitter.emit("e"), 1, "outer emit ran the once entry");
  assert.equal(count, 1, "no reentrant second run");
}

// a nested emit consumes a later once entry; the outer emit skips it
{
  const emitter = new Emitter();
  const calls = [];
  emitter.on("e", () => {
    calls.push("driver");
    if (calls.length === 1) emitter.emit("e");
  });
  emitter.once("e", () => calls.push("late-once"));
  const ran = emitter.emit("e");
  assert.deepEqual(
    calls,
    ["driver", "driver", "late-once"],
    "nested emit ran the once entry first",
  );
  assert.equal(ran, 1, "outer emit skipped the used once entry");
}

// events are independent
{
  const emitter = new Emitter();
  const calls = [];
  emitter.on("a", () => calls.push("a"));
  emitter.on("b", () => calls.push("b"));
  emitter.emit("a");
  assert.deepEqual(calls, ["a"], "only event a ran");
}

// emit passes every argument
{
  const emitter = new Emitter();
  let seen = null;
  emitter.on("e", (...args) => (seen = args));
  emitter.emit("e", 1, "two", [3]);
  assert.deepEqual(seen, [1, "two", [3]], "all arguments arrive");
}

console.log("all 12 sections passed");
