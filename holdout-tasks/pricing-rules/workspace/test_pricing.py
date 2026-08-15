import unittest

from pricing import DiscountRule, price_after_discount


class PriceAfterDiscountTests(unittest.TestCase):
    def test_no_rules(self) -> None:
        self.assertEqual(price_after_discount(250, 3, ()), 750)

    def test_no_valid_rule(self) -> None:
        rules = (DiscountRule(5, 100),)
        self.assertEqual(price_after_discount(250, 3, rules), 750)

    def test_rule_applies_at_exact_quantity(self) -> None:
        rules = (DiscountRule(3, 125),)
        self.assertEqual(price_after_discount(250, 3, rules), 625)

    def test_largest_valid_discount_wins(self) -> None:
        rules = (
            DiscountRule(2, 75),
            DiscountRule(3, 200),
            DiscountRule(8, 600),
            DiscountRule(1, 50),
        )
        self.assertEqual(price_after_discount(300, 4, rules), 1000)

    def test_discount_cannot_make_price_negative(self) -> None:
        rules = (DiscountRule(1, 900),)
        self.assertEqual(price_after_discount(200, 2, rules), 0)

    def test_zero_quantity(self) -> None:
        rules = (DiscountRule(0, 100),)
        self.assertEqual(price_after_discount(200, 0, rules), 0)

    def test_zero_discount(self) -> None:
        rules = (DiscountRule(1, 0),)
        self.assertEqual(price_after_discount(99, 4, rules), 396)

    def test_rules_do_not_change(self) -> None:
        rules = [DiscountRule(2, 100), DiscountRule(4, 300)]
        original = list(rules)
        price_after_discount(100, 4, rules)
        self.assertEqual(rules, original)


if __name__ == "__main__":
    unittest.main()
