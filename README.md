# wti-oil-price-analysis-2026

Analyze daily West Texas Intermediate (WTI) crude oil spot prices from FRED with either an interactive notebook or an automated PDF report.

## Generate a PDF report

Requires Python 3.10+. Install the dependencies and generate the latest three-year report:

```bash
python -m pip install -r requirements.txt
python wti_report/generate.py
```

The script writes `reports/wti_analysis_YYYY-MM-DD.pdf` by default, with a price and moving-average chart, return and risk summary, drawdown, rolling 20/60-observation volatility, and daily returns plotted against historical 95% Value at Risk (VaR) and expected shortfall thresholds. The summary includes the largest daily gain and loss. Use `--output /path/to/report.pdf` to choose another output path or `--lookback-years 5` to change the analysis window. The same date's report is overwritten on a rerun.

VaR and expected shortfall are historical estimates calculated from the selected sample's daily percentage returns; they are descriptive risk measures, not forecasts or guarantees.

To run it daily at 07:00 with cron, replace `/path/to/repo` with the absolute repository path and use the Python interpreter where the requirements are installed:

```cron
0 7 * * * mkdir -p /path/to/repo/reports && /path/to/repo/.venv/bin/python /path/to/repo/wti_report/generate.py >> /path/to/repo/reports/cron.log 2>&1
```

FRED's public `DCOILWTICO` series requires no API key, though its latest observation may be delayed. The script creates the output directory when needed and reports failures to the command line/log.

## Explore interactively

The original notebook remains available at [notebooks/wti_analysis.ipynb](notebooks/wti_analysis.ipynb). To run it interactively, install the optional notebook dependencies and launch Jupyter:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/wti_analysis.ipynb
```
