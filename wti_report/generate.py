#!/usr/bin/env python3
"""Generate a PDF report for WTI prices, crude benchmarks, ETFs, and a baseline forecast."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
WTI_SERIES_ID = "DCOILWTICO"
CRUDE_BENCHMARKS = {"Brent": "DCOILBRENTEU"}
ETF_TICKERS = ("USO", "BNO")
FORECAST_HORIZONS = (30, 60)
TRADING_DAYS_PER_YEAR = 252
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fetch_fred_data(
    series_id: str,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch daily observations for a FRED series over an inclusive date range."""
    end = pd.Timestamp(end_date or date.today()).normalize()
    start = pd.Timestamp(start_date).normalize() if start_date else end - pd.DateOffset(years=3)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    query = urlencode(
        {"id": series_id, "cosd": start.strftime("%Y-%m-%d"), "coed": end.strftime("%Y-%m-%d")}
    )
    raw = pd.read_csv(f"{FRED_CSV_URL}?{query}")
    if raw.empty or series_id not in raw.columns:
        raise ValueError(f"FRED returned no {series_id} observations for the requested range")

    prices = raw.rename(columns={raw.columns[0]: "date", series_id: "price"})[["date", "price"]]
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["price"]).drop_duplicates(subset="date", keep="last")
    prices = prices.set_index("date").sort_index()
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if prices.empty:
        raise ValueError(f"No usable {series_id} observations found in the requested date range")
    return prices


def fetch_wti_data(
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch daily WTI spot prices from FRED for an inclusive date range."""
    return fetch_fred_data(WTI_SERIES_ID, start_date, end_date)


def fetch_etf_data(
    ticker: str,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch daily adjusted close prices from Yahoo Finance's chart endpoint."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    params = urlencode({
        "period1": int(start.timestamp()),
        "period2": int((end + pd.Timedelta(days=1)).timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    })
    request = Request(
        f"{YAHOO_CHART_URL.format(ticker=ticker)}?{params}",
        headers={"User-Agent": "Mozilla/5.0 (compatible; WTIPriceReport/1.0)"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result")
    if not result:
        message = payload.get("chart", {}).get("error") or "no price observations"
        raise ValueError(f"Yahoo Finance returned no {ticker} data: {message}")
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    adjusted = chart.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    values = adjusted if adjusted is not None else quote.get("close", [])
    prices = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        "price": values,
    })
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["price"]).set_index("date").sort_index()
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if prices.empty:
        raise ValueError(f"No usable {ticker} adjusted-close observations in the requested date range")
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
    analysis["volatility_20d_annualized"] = (
        analysis["daily_return"].rolling(20, min_periods=20).std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    analysis["drawdown"] = analysis["price"] / analysis["price"].cummax() - 1
    return analysis


def baseline_forecast(
    analysis: pd.DataFrame, horizons: Sequence[int] = FORECAST_HORIZONS
) -> pd.DataFrame:
    """Return a naïve random-walk forecast: hold the latest observed price constant."""
    if analysis.empty or "price" not in analysis:
        raise ValueError("analysis must contain at least one 'price' observation")
    if any(not isinstance(days, (int, np.integer)) or days < 1 for days in horizons):
        raise ValueError("forecast horizons must be positive whole numbers of trading days")
    latest_date = pd.Timestamp(analysis.index[-1])
    latest_price = float(analysis["price"].iloc[-1])
    forecast_dates = pd.bdate_range(latest_date + pd.offsets.BDay(1), periods=max(horizons))
    return pd.DataFrame({
        "horizon_trading_days": list(horizons),
        "date": [forecast_dates[days - 1] for days in horizons],
        "forecast_price": [latest_price] * len(horizons),
    }).set_index("horizon_trading_days")


def summarize_analysis(analysis: pd.DataFrame) -> dict[str, object]:
    """Calculate period-level summary metrics for an analysis frame."""
    if len(analysis) < 2:
        raise ValueError("at least two analyzed observations are required")

    returns = analysis["daily_return"].dropna()
    if len(returns) < 2:
        raise ValueError("at least three price observations are required to calculate volatility")
    return {
        "start_date": analysis.index.min().date(),
        "end_date": analysis.index.max().date(),
        "observations": len(analysis),
        "starting_price": float(analysis["price"].iloc[0]),
        "latest_price": float(analysis["price"].iloc[-1]),
        "cumulative_return": float(analysis["return_index"].iloc[-1] - 1),
        "annualized_volatility": float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "maximum_drawdown": float(analysis["drawdown"].min()),
    }


def _add_report_header(fig: plt.Figure, title: str, generated_at: datetime) -> None:
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.94,
        f"FRED series {WTI_SERIES_ID} | Generated {generated_at:%Y-%m-%d %H:%M UTC}",
        ha="center", fontsize=9, color="dimgray",
    )
    fig.text(
        0.5, 0.015,
        "Source: Federal Reserve Bank of St. Louis, FRED. ETF prices: Yahoo Finance.",
        ha="center", fontsize=8, color="dimgray",
    )


def generate_pdf_report(
    analysis: pd.DataFrame,
    output_path: str | Path,
    comparison_prices: dict[str, pd.DataFrame] | None = None,
    forecast_horizons: Sequence[int] = FORECAST_HORIZONS,
) -> Path:
    """Write a PDF report with WTI analysis, comparisons, and baseline forecasts."""
    metrics = summarize_analysis(analysis)
    forecasts = baseline_forecast(analysis, forecast_horizons)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)

    with PdfPages(output, metadata={
        "Title": "WTI Crude Oil Price Analysis",
        "Author": "WTI Price Analysis",
        "Subject": f"FRED {WTI_SERIES_ID} report with ETF comparisons and baseline forecast",
    }) as pdf:
        summary_fig = plt.figure(figsize=(11.7, 8.3))
        _add_report_header(summary_fig, "WTI CRUDE OIL PRICE REPORT", generated_at)
        grid = summary_fig.add_gridspec(2, 1, left=0.08, right=0.96, top=0.89, bottom=0.08, height_ratios=(1, 3))

        table_axis = summary_fig.add_subplot(grid[0])
        table_axis.axis("off")
        table_rows = [
            ("Reporting period", f"{metrics['start_date']} to {metrics['end_date']}"),
            ("Observations", f"{metrics['observations']:,} daily prices"),
            ("Starting / latest price", f"${metrics['starting_price']:,.2f} / ${metrics['latest_price']:,.2f}"),
            ("Cumulative return", f"{metrics['cumulative_return']:.2%}"),
            ("Annualized daily-return volatility", f"{metrics['annualized_volatility']:.2%}"),
            ("Maximum drawdown", f"{metrics['maximum_drawdown']:.2%}"),
        ]
        table = table_axis.table(cellText=table_rows, colLabels=("Metric", "Value"), colWidths=(0.42, 0.58), cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)

        price_axis = summary_fig.add_subplot(grid[1])
        price_axis.plot(analysis.index, analysis["price"], label="WTI spot price", linewidth=1.5)
        for window in (20, 50, 200):
            price_axis.plot(analysis.index, analysis[f"ma_{window}"], label=f"{window}-observation average", linewidth=1)
        price_axis.set_title("Daily spot price and moving averages")
        price_axis.set_ylabel("USD per barrel")
        price_axis.grid(alpha=0.25)
        price_axis.legend(loc="best", ncol=2)
        summary_fig.autofmt_xdate()
        pdf.savefig(summary_fig)
        plt.close(summary_fig)

        risk_fig, (drawdown_axis, volatility_axis) = plt.subplots(2, 1, figsize=(11.7, 8.3), sharex=True)
        _add_report_header(risk_fig, "WTI RISK INDICATORS", generated_at)
        risk_fig.subplots_adjust(top=0.88, bottom=0.09, hspace=0.3)
        drawdown_axis.fill_between(analysis.index, analysis["drawdown"] * 100, 0, alpha=0.35, color="firebrick")
        drawdown_axis.set_title("Drawdown from the previous price peak")
        drawdown_axis.set_ylabel("Drawdown (%)")
        drawdown_axis.grid(alpha=0.25)
        volatility_axis.plot(analysis.index, analysis["volatility_20d_annualized"] * 100, color="darkorange", linewidth=1.2)
        volatility_axis.set_title("20-observation annualized volatility")
        volatility_axis.set_ylabel("Volatility (%)")
        volatility_axis.set_xlabel("Date")
        volatility_axis.grid(alpha=0.25)
        risk_fig.autofmt_xdate()
        pdf.savefig(risk_fig)
        plt.close(risk_fig)

        forecast_fig, (comparison_axis, forecast_axis) = plt.subplots(2, 1, figsize=(11.7, 8.3))
        _add_report_header(forecast_fig, "BENCHMARKS, ETFS & BASELINE FORECAST", generated_at)
        forecast_fig.subplots_adjust(top=0.88, bottom=0.1, hspace=0.38)
        comparison_series = {"WTI": analysis[["price"]], **(comparison_prices or {})}
        for label, frame in comparison_series.items():
            if frame.empty or "price" not in frame:
                continue
            series = frame["price"].dropna().sort_index()
            if series.empty:
                continue
            comparison_axis.plot(series.index, (series / series.iloc[0] - 1) * 100, label=label, linewidth=1.2)
        comparison_axis.set_title("Crude benchmarks and ETF adjusted-close performance")
        comparison_axis.set_ylabel("Change from period start (%)")
        comparison_axis.grid(alpha=0.25)
        comparison_axis.legend(loc="best", ncol=3)

        recent = analysis.tail(90)
        forecast_axis.plot(recent.index, recent["price"], label="WTI historical price", color="steelblue")
        latest_date = pd.Timestamp(analysis.index[-1])
        latest_price = float(analysis["price"].iloc[-1])
        future_dates = pd.bdate_range(latest_date + pd.offsets.BDay(1), periods=max(forecast_horizons))
        forecast_axis.plot([latest_date, future_dates[-1]], [latest_price, latest_price], linestyle="--", color="darkorange", label="Baseline: last price held constant")
        for horizon, row in forecasts.iterrows():
            forecast_axis.scatter(row["date"], row["forecast_price"], zorder=3)
            forecast_axis.annotate(f"{horizon}-trading-day: ${row['forecast_price']:,.2f}", (row["date"], row["forecast_price"]), xytext=(5, 8), textcoords="offset points", fontsize=8)
        forecast_axis.set_title("WTI naïve baseline forecast (not investment advice)")
        forecast_axis.set_ylabel("USD per barrel")
        forecast_axis.set_xlabel("Date")
        forecast_axis.grid(alpha=0.25)
        forecast_axis.legend(loc="best")
        forecast_fig.autofmt_xdate()
        pdf.savefig(forecast_fig)
        plt.close(forecast_fig)

    return output.resolve()


def _positive_int(value: str) -> int:
    years = int(value)
    if years < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return years


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch daily WTI prices and generate a PDF analysis, benchmark, ETF, and forecast report.")
    parser.add_argument("--lookback-years", type=_positive_int, default=3, help="number of years of price history to include (default: 3)")
    parser.add_argument("--output", type=Path, help="PDF output path (default: reports/wti_analysis_YYYY-MM-DD.pdf)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    end_date = date.today()
    start_date = pd.Timestamp(end_date) - pd.DateOffset(years=args.lookback_years)
    prices = fetch_wti_data(start_date=start_date, end_date=end_date)
    analysis = analyze_prices(prices)
    comparisons: dict[str, pd.DataFrame] = {}
    for label, series_id in CRUDE_BENCHMARKS.items():
        try:
            comparisons[label] = fetch_fred_data(series_id, start_date, end_date)
        except Exception as error:
            print(f"Warning: could not fetch {label} benchmark: {error}")
    for ticker in ETF_TICKERS:
        try:
            comparisons[ticker] = fetch_etf_data(ticker, start_date, end_date)
        except Exception as error:
            print(f"Warning: could not fetch {ticker} ETF prices: {error}")
    output = args.output or (PROJECT_ROOT / "reports" / f"wti_analysis_{end_date:%Y-%m-%d}.pdf")
    report_path = generate_pdf_report(analysis, output, comparisons)
    print(f"Created {report_path} ({len(analysis):,} observations, {analysis.index.min().date()} through {analysis.index.max().date()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
