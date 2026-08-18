// A strict CSV parser.
//
// parseCsv(text) returns an array of records; a record is an array
// of field strings. The rules are:
//
// - Fields are separated by commas. Records are separated by LF or
//   CRLF. Both separators can mix in one input.
// - An empty input gives []. A separator at the end of the input
//   terminates the last record; it does not start an empty record.
//   So "a\n" gives [["a"]] and "\n" gives [[""]].
// - A field that starts with a double quote is quoted. Inside a
//   quoted field, commas, LF, CRLF, and lone CR are literal text. A
//   pair of double quotes is one literal double quote.
// - The closing quote must be followed by a comma, a record
//   separator, or the end of the input. Anything else throws
//   Error("unexpected character after closing quote").
// - An input that ends inside a quoted field throws
//   Error("unterminated quoted field").
// - In an unquoted field a double quote after the first character
//   is a literal character. A lone CR (not followed by LF) in an
//   unquoted field is also a literal character.

export function parseCsv(text) {
  // Not implemented yet. This is the task.
  throw new Error("Not implemented");
}
