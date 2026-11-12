# wti-oil-price-analysis-2026

Analyze the latest three years of daily West Texas Intermediate (WTI) crude oil spot prices from FRED. The notebook calculates daily price changes and returns, moving averages, annualized volatility, drawdowns, summary statistics, and charts.

## Run the notebook

Requires Python 3. Install the dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/wti_analysis.ipynb
```

The notebook downloads the public FRED series `DCOILWTICO` when run; no API key is required. FRED data may be delayed, so the latest observation can be earlier than today. The date range and analysis functions can be adjusted in the notebook.
