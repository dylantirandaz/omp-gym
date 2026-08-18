import assert from "node:assert/strict";

import { createRouter, splitPath } from "./router.mjs";

assert.deepEqual(splitPath("/"), [], "root path has no segments");
assert.deepEqual(splitPath("/a/b"), ["a", "b"], "plain split");
assert.deepEqual(splitPath("/a/"), ["a"], "trailing slash drops out");

// root and static routes
{
  const router = createRouter();
  router.register("/", "home");
  router.register("/about", "about");
  assert.deepEqual(router.resolve("/"), { handler: "home", params: {} }, "root");
  assert.deepEqual(
    router.resolve("/about"),
    { handler: "about", params: {} },
    "static route",
  );
  assert.equal(router.resolve("/missing"), null, "unknown path");
  assert.equal(router.resolve("/about/deep"), null, "path too deep");
}

// params
{
  const router = createRouter();
  router.register("/users/:id", "user");
  router.register("/users/:id/posts/:post", "post");
  assert.deepEqual(
    router.resolve("/users/7"),
    { handler: "user", params: { id: "7" } },
    "one param",
  );
  assert.deepEqual(
    router.resolve("/users/7/posts/9"),
    { handler: "post", params: { id: "7", post: "9" } },
    "two params",
  );
  assert.equal(router.resolve("/users"), null, "param needs a segment");
  assert.deepEqual(
    router.resolve("/users/8"),
    { handler: "user", params: { id: "8" } },
    "params never leak between resolves",
  );
}

// precedence: static beats param beats wildcard
{
  const router = createRouter();
  router.register("/files/readme", "static");
  router.register("/files/:name", "param");
  router.register("/files/*rest", "wild");
  assert.deepEqual(
    router.resolve("/files/readme"),
    { handler: "static", params: {} },
    "static wins",
  );
  assert.deepEqual(
    router.resolve("/files/other"),
    { handler: "param", params: { name: "other" } },
    "param beats wildcard",
  );
  assert.deepEqual(
    router.resolve("/files/a/b/c"),
    { handler: "wild", params: { rest: "a/b/c" } },
    "wildcard joins the rest",
  );
  assert.equal(router.resolve("/files"), null, "wildcard needs a segment");
}

// backtracking out of a static dead end
{
  const router = createRouter();
  router.register("/a/b", "short-static");
  router.register("/a/:x/c", "param-deep");
  assert.deepEqual(
    router.resolve("/a/b"),
    { handler: "short-static", params: {} },
    "static leaf still wins",
  );
  assert.deepEqual(
    router.resolve("/a/b/c"),
    { handler: "param-deep", params: { x: "b" } },
    "search backtracks from static to param",
  );
}

// backtracking down to the wildcard
{
  const router = createRouter();
  router.register("/a/:x", "param");
  router.register("/a/*rest", "wild");
  assert.deepEqual(
    router.resolve("/a/b/c"),
    { handler: "wild", params: { rest: "b/c" } },
    "param dead end falls back to wildcard",
  );
}

// registration errors
{
  const router = createRouter();
  router.register("/dup", "one");
  assert.throws(
    () => router.register("/dup", "two"),
    { message: "duplicate route" },
    "duplicate static route",
  );
  router.register("/w/*rest", "w1");
  assert.throws(
    () => router.register("/w/*other", "w2"),
    { message: "duplicate route" },
    "duplicate wildcard",
  );
  router.register("/p/:id", "p1");
  assert.throws(
    () => router.register("/p/:name", "p2"),
    { message: "conflicting param names" },
    "conflicting param names",
  );
  assert.throws(
    () => router.register("/x/*rest/tail", "x"),
    { message: "wildcard must be the last segment" },
    "wildcard must be last",
  );
}

// same param name on two routes shares the node
{
  const router = createRouter();
  router.register("/p/:id", "leaf");
  router.register("/p/:id/edit", "edit");
  assert.deepEqual(
    router.resolve("/p/5/edit"),
    { handler: "edit", params: { id: "5" } },
    "shared param node",
  );
}

console.log("all 8 sections passed");
