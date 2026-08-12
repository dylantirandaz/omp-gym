import sys

from report import total_column

failures = 0

units = total_column("data.csv", "units")
if units != 55:
    print(f"FAIL: units total = {units!r}, expected 55")
    failures += 1

price = total_column("data.csv", "price")
if abs(price - 44.75) > 1e-9:
    print(f"FAIL: price total = {price!r}, expected 44.75")
    failures += 1

if failures:
    sys.exit(1)
print("all 2 cases passed")
