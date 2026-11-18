#!/usr/bin/env python3
"""Generate a PDF report for the latest daily WTI crude oil prices."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
WTI_SERIES_ID = "DCOILWTICO"
TRADING_DAYS_PER_YEAR = 252
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fetch_wti_data(
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch daily WTI spot prices from FRED for an inclusive date range."""
    end = pd.Timestamp(end_date or date.today()).normalize()
    start = pd.Timestamp(start_date).normalize() if start_date else end - pd.DateOffset(years=3)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    query = urlencode(
        {
            "id": WTI_SERIES_ID,
            "cosd": start.strftime("%Y-%m-%d"),
            "coed": end.strftime("%Y-%m-%d"),
        }
    )
    raw = pd.read_csv(f"{FRED_CSV_URL}?{query}")
    if raw.empty or WTI_SERIES_ID not in raw.columns:
        raise ValueError(f"FRED returned no {WTI_SERIES_ID} observations for the requested range")

    date_column = raw.columns[0]
    prices = raw.rename(
        columns={date_column: "date", WTI_SERIES_ID: "price"}
    )[["date", "price"]]
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["price"]).drop_duplicates(subset="date", keep="last")
    prices = prices.set_index("date").sort_index()
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]

    if prices.empty:
        raise ValueError(f"No usable {WTI_SERIES_ID} observations found in the requested date range")
    return prices


def analyze_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns, moving averages, volatility, and drawdowns."""
    if "price" not in prices.columns:
        raise ValueError("prices must contain a 'price' column")
    if len(prices) < 2:
        raise ValueError("at least two price observations are required")

    analysis = prices[["price"]].copy().sort_index()
    values = analysis["price"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and positive for percentage-return analysis")

    analysis["daily_change"] = analysis["price"].diff()
    analysis["daily_return"] = analysis["price"].pct_change(fill_method=None)
    analysis["return_index"] = analysis["price"] / analysis["price"].iloc[0]
    for window in (20, 50, 200):
        analysis[f"ma_{window}"] = analysis["price"].rolling(window, min_periods=window).mean()
    for window in (20, 60):
        analysis[f"volatility_{window}d_annualized"] = (
            analysis["daily_return"].rolling(window, min_periods=window).std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    analysis["drawdown"] = analysis["price"] / analysis["price"].cummax() - 1
    return analysis


def summarize_analysis(analysis: pd.DataFrame) -> dict[str, object]:
    """Calculate period-level summary and historical tail-risk metrics."""
    if len(analysis) < 2:
        raise ValueError("at least two analyzed observations are required")

    returns = analysis["daily_return"].dropna()
    if len(returns) < 2:
        raise ValueError("at least three price observations are required to calculate volatility")

    var_threshold = float(returns.quantile(0.05))
    tail_returns = returns[returns <= var_threshold]
    return {
        "start_date": analysis.index.min().date(),
        "end_date": analysis.index.max().date(),
        "observations": len(analysis),
        "starting_price": float(analysis["price"].iloc[0]),
        "latest_price": float(analysis["price"].iloc[-1]),
        "cumulative_return": float(analysis["return_index"].iloc[-1] - 1),
        "annualized_volatility": float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "maximum_drawdown": float(analysis["drawdown"].min()),
        # Report risk measures as non-negative loss percentages, even if the
        # empirical return quantile happens to be positive in a tiny sample.
        "var_95": max(0.0, -var_threshold),
        "expected_shortfall_95": max(0.0, float(-tail_returns.mean())),
        "largest_daily_gain": float(returns.max()),
        "largest_daily_loss": float(returns.min()),
    }


def _add_report_header(fig: plt.Figure, title: str, generated_at: datetime) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.94,
        f"FRED series {WTI_SERIES_ID} | Generated {generated_at:%Y-%m-%d %H:%M UTC}",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.text(
        0.5,
        0.015,
        "Source: Federal Reserve Bank of St. Louis, FRED. Prices in USD per barrel.",
        ha="center",
        fontsize=8,
        color="dimgray",
    )


def generate_pdf_report(analysis: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a two-page PDF report and return its path."""
    metrics = summarize_analysis(analysis)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)

    with PdfPages(
        output,
        metadata={
            "Title": "WTI Crude Oil Price Analysis",
            "Author": "WTI Price Analysis",
            "Subject": f"FRED {WTI_SERIES_ID} daily price report",
        },
    ) as pdf:
        summary_fig = plt.figure(figsize=(11.7, 8.3))
        _add_report_header(summary_fig, "WTI CRUDE OIL PRICE REPORT", generated_at)
        grid = summary_fig.add_gridspec(
            2, 1, left=0.08, right=0.96, top=0.89, bottom=0.08, height_ratios=(1, 3)
        )

        table_axis = summary_fig.add_subplot(grid[0])
        table_axis.axis("off")
        table_rows = [
            ("Reporting period", f"{metrics['start_date']} to {metrics['end_date']}"),
            ("Observations", f"{metrics['observations']:,} daily prices"),
            ("Starting / latest price", f"${metrics['starting_price']:,.2f} / ${metrics['latest_price']:,.2f}"),
            ("Cumulative return", f"{metrics['cumulative_return']:.2%}"),
            ("Annualized daily-return volatility", f"{metrics['annualized_volatility']:.2%}"),
            ("Maximum drawdown", f"{metrics['maximum_drawdown']:.2%}"),
            ("95% daily VaR (historical)", f"{metrics['var_95']:.2%}"),
            ("95% daily expected shortfall", f"{metrics['expected_shortfall_95']:.2%}"),
            ("Largest daily gain / loss", f"{metrics['largest_daily_gain']:.2%} / {metrics['largest_daily_loss']:.2%}"),
        ]
        table = table_axis.table(
            cellText=table_rows,
            colLabels=("Metric", "Value"),
            colWidths=(0.42, 0.58),
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.15)

        price_axis = summary_fig.add_subplot(grid[1])
        price_axis.plot(analysis.index, analysis["price"], label="WTI spot price", linewidth=1.5)
        for window in (20, 50, 200):
            price_axis.plot(
                analysis.index,
                analysis[f"ma_{window}"],
                label=f"{window}-observation average",
                linewidth=1,
            )
        price_axis.set_title("Daily spot price and moving averages")
        price_axis.set_ylabel("USD per barrel")
        price_axis.grid(alpha=0.25)
        price_axis.legend(loc="best", ncol=2)
        summary_fig.autofmt_xdate()
        pdf.savefig(summary_fig)
        plt.close(summary_fig)

        risk_fig, (drawdown_axis, volatility_axis, returns_axis) = plt.subplots(
            3, 1, figsize=(11.7, 8.3), sharex=True,
            gridspec_kw={"height_ratios": (1.2, 1, 1)},
        )
        _add_report_header(risk_fig, "WTI RISK INDICATORS", generated_at)
        risk_fig.subplots_adjust(top=0.88, bottom=0.09, hspace=0.42)

        drawdown_axis.fill_between(
            analysis.index, analysis["drawdown"] * 100, 0, alpha=0.35, color="firebrick"
        )
        drawdown_axis.set_title("Drawdown from the previous price peak")
        drawdown_axis.set_ylabel("Drawdown (%)")
        drawdown_axis.grid(alpha=0.25)

        volatility_axis.plot(
            analysis.index,
            analysis["volatility_20d_annualized"] * 100,
            label="20-observation",
            color="darkorange",
            linewidth=1.2,
        )
        volatility_axis.plot(
            analysis.index,
            analysis["volatility_60d_annualized"] * 100,
            label="60-observation",
            color="steelblue",
            linewidth=1.2,
        )
        volatility_axis.set_title("Rolling annualized volatility")
        volatility_axis.set_ylabel("Volatility (%)")
        volatility_axis.legend(loc="best")
        volatility_axis.grid(alpha=0.25)

        daily_var = -metrics["var_95"] * 100
        daily_es = -metrics["expected_shortfall_95"] * 100
        returns_axis.plot(analysis.index, analysis["daily_return"] * 100, color="slategray", linewidth=0.8)
        returns_axis.axhline(daily_var, color="darkorange", linestyle="--", label="95% VaR threshold")
        returns_axis.axhline(daily_es, color="firebrick", linestyle=":", label="95% expected shortfall")
        returns_axis.set_title("Daily returns and historical downside thresholds")
        returns_axis.set_ylabel("Daily return (%)")
        returns_axis.set_xlabel("Date")
        returns_axis.legend(loc="best")
        returns_axis.grid(alpha=0.25)

        risk_fig.autofmt_xdate()
        pdf.savefig(risk_fig)
        plt.close(risk_fig)

    return output.resolve()


def _positive_int(value: str) -> int:
    years = int(value)
    if years < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return years


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch daily WTI prices from FRED and generate a PDF analysis report."
    )
    parser.add_argument(
        "--lookback-years",
        type=_positive_int,
        default=3,
        help="number of years of price history to include (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PDF output path (default: reports/wti_analysis_YYYY-MM-DD.pdf)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    end_date = date.today()
    start_date = pd.Timestamp(end_date) - pd.DateOffset(years=args.lookback_years)
    prices = fetch_wti_data(start_date=start_date, end_date=end_date)
    analysis = analyze_prices(prices)
    output = args.output or (
        PROJECT_ROOT / "reports" / f"wti_analysis_{end_date:%Y-%m-%d}.pdf"
    )
    report_path = generate_pdf_report(analysis, output)
    print(
        f"Created {report_path} "
        f"({len(analysis):,} observations, "
        f"{analysis.index.min().date()} through {analysis.index.max().date()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
