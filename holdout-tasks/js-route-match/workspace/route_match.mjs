/**
 * Match one path against a route pattern.
 *
 * Both inputs start with a slash. A pattern segment that starts with a colon
 * captures one path segment. Decode captured values. Ignore one trailing
 * slash, but do not ignore an empty segment inside a path. Each parameter
 * name occurs once. Return { matched: false } when the route does not match.
 * Return { matched: true, params } when the route matches.
 */
export function matchRoute(pattern, path) {
  throw new Error("Not implemented");
}
