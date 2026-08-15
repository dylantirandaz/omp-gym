import assert from "node:assert/strict";

import { LruCache } from "./lru_cache.mjs";

assert.throws(() => new LruCache(0), RangeError, "zero capacity");
assert.throws(() => new LruCache(1.5), RangeError, "fractional capacity");

{
  const cache = new LruCache(2);
  assert.equal(cache.size(), 0, "new cache size");
  assert.deepEqual(cache.get("missing"), { found: false }, "cache miss");
}

{
  const cache = new LruCache(2);
  assert.deepEqual(cache.put("a", 10), { evicted: false }, "first put");
  assert.deepEqual(cache.get("a"), { found: true, value: 10 }, "cache hit");
  assert.equal(cache.size(), 1, "size after one put");
}

{
  const cache = new LruCache(2);
  cache.put("a", 1);
  cache.put("b", 2);
  assert.deepEqual(cache.get("a"), { found: true, value: 1 });
  assert.deepEqual(
    cache.put("c", 3),
    { evicted: true, key: "b", value: 2 },
    "get updates recent use",
  );
  assert.deepEqual(cache.get("b"), { found: false });
  assert.deepEqual(cache.get("a"), { found: true, value: 1 });
  assert.deepEqual(cache.get("c"), { found: true, value: 3 });
}

{
  const cache = new LruCache(2);
  cache.put("a", 1);
  cache.put("b", 2);
  assert.deepEqual(cache.put("a", 7), { evicted: false }, "update result");
  assert.equal(cache.size(), 2, "update does not add an entry");
  assert.deepEqual(
    cache.put("c", 3),
    { evicted: true, key: "b", value: 2 },
    "update refreshes recent use",
  );
  assert.deepEqual(cache.get("a"), { found: true, value: 7 });
}

{
  const cache = new LruCache(1);
  cache.put("x", undefined);
  assert.deepEqual(
    cache.get("x"),
    { found: true, value: undefined },
    "undefined value is present",
  );
  assert.deepEqual(
    cache.put("y", 2),
    { evicted: true, key: "x", value: undefined },
    "capacity one eviction",
  );
  assert.equal(cache.size(), 1);
}

console.log("all LruCache cases passed");
