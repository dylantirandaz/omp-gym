import sys

from fizzbuzz import fizzbuzz

CASES = [
    (1, "1"),
    (2, "2"),
    (3, "Fizz"),
    (5, "Buzz"),
    (9, "Fizz"),
    (10, "Buzz"),
    (15, "FizzBuzz"),
    (30, "FizzBuzz"),
    (45, "FizzBuzz"),
    (7, "7"),
]

failures = 0
for value, expected in CASES:
    actual = fizzbuzz(value)
    if actual != expected:
        print(f"FAIL: fizzbuzz({value}) = {actual!r}, expected {expected!r}")
        failures += 1

if failures:
    print(f"{failures} of {len(CASES)} cases failed")
    sys.exit(1)
print(f"all {len(CASES)} cases passed")
