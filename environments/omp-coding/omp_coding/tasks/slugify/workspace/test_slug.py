import sys

from slug import slugify

CASES = [
    ("Hello World", "hello-world"),
    ("  spaces  everywhere  ", "spaces-everywhere"),
    ("snake_case_title", "snake-case-title"),
    ("Already-Hyphenated", "already-hyphenated"),
    ("Symbols! & Junk?", "symbols-junk"),
    ("MiXeD CaSe 123", "mixed-case-123"),
    ("--edge--case--", "edge-case"),
    ("", ""),
]

failures = 0
for title, expected in CASES:
    actual = slugify(title)
    if actual != expected:
        print(f"FAIL: slugify({title!r}) = {actual!r}, expected {expected!r}")
        failures += 1

if failures:
    print(f"{failures} of {len(CASES)} cases failed")
    sys.exit(1)
print(f"all {len(CASES)} cases passed")
