/** A least-recently-used cache with a fixed positive capacity. */
export class LruCache {
  #capacity;

  constructor(capacity) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError("capacity must be a positive integer");
    }
    this.#capacity = capacity;
  }

  /** Return { found: false } or { found: true, value }. */
  get(key) {
    throw new Error("Not implemented");
  }

  /**
   * Add or update one value.
   *
   * Return { evicted: false } when no entry is removed. Return
   * { evicted: true, key, value } for the removed least-recently-used entry.
   */
  put(key, value) {
    throw new Error("Not implemented");
  }

  size() {
    throw new Error("Not implemented");
  }
}
