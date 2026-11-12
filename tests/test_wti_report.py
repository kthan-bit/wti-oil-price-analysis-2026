import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from wti_report import generate


class WtiReportTests(unittest.TestCase):
    def test_fetch_wti_data_cleans_missing_values_and_requests_dates(self):
        response = pd.DataFrame(
            {
                "observation_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "DCOILWTICO": ["70.0", "71.0", "."],
            }
        )
        with patch.object(generate.pd, "read_csv", return_value=response) as read_csv:
            prices = generate.fetch_wti_data("2024-01-01", "2024-01-10")

        requested_url = read_csv.call_args.args[0]
        self.assertIn("cosd=2024-01-01", requested_url)
        self.assertIn("coed=2024-01-10", requested_url)
        self.assertEqual(len(prices), 2)
        self.assertEqual(prices.index.name, "date")
        self.assertEqual(prices["price"].tolist(), [70.0, 71.0])

    def test_analyze_and_summarize_prices(self):
        prices = pd.DataFrame(
            {"price": [100.0, 110.0, 99.0, 108.9]},
            index=pd.date_range("2024-01-01", periods=4),
        )

        analysis = generate.analyze_prices(prices)
        summary = generate.summarize_analysis(analysis)

        self.assertAlmostEqual(summary["cumulative_return"], 0.089)
        self.assertAlmostEqual(summary["maximum_drawdown"], -0.1)
        self.assertEqual(summary["observations"], 4)

    def test_analysis_rejects_non_positive_prices(self):
        prices = pd.DataFrame(
            {"price": [10.0, 0.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            generate.analyze_prices(prices)

    def test_summary_requires_enough_observations_for_volatility(self):
        prices = pd.DataFrame(
            {"price": [10.0, 11.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        with self.assertRaisesRegex(ValueError, "three price observations"):
            generate.summarize_analysis(generate.analyze_prices(prices))

    def test_report_writes_pdf(self):
        prices = pd.DataFrame(
            {"price": [100 + index * 0.2 for index in range(220)]},
            index=pd.date_range("2024-01-01", periods=220),
        )
        analysis = generate.analyze_prices(prices)

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "report.pdf"
            result = generate.generate_pdf_report(analysis, output_path)

            self.assertEqual(result, output_path.resolve())
            self.assertTrue(result.read_bytes().startswith(b"%PDF"))
            self.assertGreater(result.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
