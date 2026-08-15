from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscountRule:
    minimum_quantity: int
    discount_cents: int


def price_after_discount(
    unit_price_cents: int,
    quantity: int,
    rules: Sequence[DiscountRule],
) -> int:
    """Return the order price after the best valid discount.

    The price values and quantity are nonnegative integers. A rule is valid
    when the quantity is not less than its minimum quantity. Use the largest
    valid discount. The result cannot be less than zero. Do not change rules.
    """
    raise NotImplementedError
