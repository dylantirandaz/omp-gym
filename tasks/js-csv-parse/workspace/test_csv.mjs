import assert from "node:assert/strict";

import { parseCsv } from "./csv.mjs";

assert.deepEqual(parseCsv(""), [], "empty input has no records");
assert.deepEqual(parseCsv("\n"), [[""]], "one empty record");
assert.deepEqual(parseCsv("a,b,c"), [["a", "b", "c"]], "simple record");
assert.deepEqual(parseCsv("a\n"), [["a"]], "trailing LF adds no record");
assert.deepEqual(parseCsv("a\r\n"), [["a"]], "trailing CRLF adds no record");
assert.deepEqual(
  parseCsv("a,b\nc,d"),
  [["a", "b"], ["c", "d"]],
  "two records",
);
assert.deepEqual(
  parseCsv("a\r\nb\nc"),
  [["a"], ["b"], ["c"]],
  "mixed CRLF and LF separators",
);
assert.deepEqual(parseCsv("a,,b"), [["a", "", "b"]], "empty middle field");
assert.deepEqual(parseCsv(",\n,"), [["", ""], ["", ""]], "all empty fields");
assert.deepEqual(
  parseCsv('"a,b",c'),
  [["a,b", "c"]],
  "comma inside a quoted field",
);
assert.deepEqual(
  parseCsv('"line1\nline2",x'),
  [["line1\nline2", "x"]],
  "LF inside a quoted field",
);
assert.deepEqual(
  parseCsv('"crlf\r\nkept"'),
  [["crlf\r\nkept"]],
  "CRLF inside a quoted field is literal",
);
assert.deepEqual(
  parseCsv('"say ""hi"""'),
  [['say "hi"']],
  "escaped quotes",
);
assert.deepEqual(parseCsv('""'), [[""]], "empty quoted field");
assert.deepEqual(
  parseCsv('"",x\n"y",""'),
  [["", "x"], ["y", ""]],
  "quoted fields at record edges",
);
assert.deepEqual(
  parseCsv('ab"c'),
  [['ab"c']],
  "mid-field quote is literal in an unquoted field",
);
assert.deepEqual(
  parseCsv("a\rb"),
  [["a\rb"]],
  "lone CR is literal in an unquoted field",
);
assert.throws(
  () => parseCsv('"open'),
  { message: "unterminated quoted field" },
  "unterminated quote throws",
);
assert.throws(
  () => parseCsv('"a"x'),
  { message: "unexpected character after closing quote" },
  "junk after a closing quote throws",
);
assert.deepEqual(
  parseCsv('"a"\n"b",c\n'),
  [["a"], ["b", "c"]],
  "closing quote before LF and comma",
);
console.log("all 20 cases passed");
