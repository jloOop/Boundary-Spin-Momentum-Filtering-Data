"""Presentation regressions: preserve decimal rounding, signs and missing values."""
import unittest
from .supplement_tables import cell


class PresentationTests(unittest.TestCase):
    def test_paper_rounding_discrepancies_are_not_hidden(self):
        self.assertEqual(cell("0.99971289084281545", {"key": "P_nc_z", "format": ".3f"}), "1.000")
        self.assertEqual(cell("1.823508", {"key": "epsilon_percent", "format": ".3f"}), "1.824")
        self.assertEqual(cell("0.739745182478", {"key": "D", "format": ".3f"}), "0.740")

    def test_decimal_ties_signs_and_scientific_notation(self):
        self.assertEqual(cell("-0.00001234", {"key": "budget", "format": ".2e"}), "-1.23e-5")
        self.assertEqual(cell("2.345", {"key": "value", "format": ".2f"}), "2.34")
        self.assertEqual(cell("2.355", {"key": "value", "format": ".2f"}), "2.36")
        self.assertEqual(cell("0.125", {"key": "w_L", "format": ".3f"}), "0.125")

    def test_missing_and_percent_tokens_cannot_become_numbers(self):
        for raw in ("", "NaN", "Infinity", "12.5%", "1,25"):
            with self.assertRaises(ValueError):
                cell(raw, {"key": "probability", "format": ".3f"})
        self.assertEqual(cell("", {"key": "unavailable", "format": ".3f", "allow_missing": True}), "—")
        with self.assertRaises(ValueError):
            cell("2999.5", {"key": "Nz", "format": "d"})

    def test_identifiers_and_delimiters_stay_in_one_cell(self):
        self.assertEqual(cell("neutral", {"key": "branch", "values": {"neutral": "neutral sector"}}), "neutral sector")
        self.assertEqual(cell("a|b\nc", {"key": "label"}), "a&#124;b c")


if __name__ == "__main__":
    unittest.main()
