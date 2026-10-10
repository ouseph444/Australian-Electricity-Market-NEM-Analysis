# Australian Electricity Market (NEM) Analysis

An end-to-end energy market analysis in Python: prices, demand, renewables, gas, battery value, hedging and demand forecasting for four National Electricity Market regions (NSW1, QLD1, SA1, VIC1).

**Open `Australian_Energy_Market_Analysis.ipynb`.** It is written to be read top to bottom, with a short explanation before every step, and it already contains all outputs and charts.

## Dataset (synthetic)

| File | Contents |
|---|---|
| `data/nem_trading_30min.csv` | Half-hourly demand, wind, utility solar, temperature and spot price (RRP) by region, 1 Jul 2024 to 30 Jun 2026 (about 140,000 rows) |
| `data/gas_price_daily.csv` | Illustrative daily east-coast gas price (A$/GJ) |
| `generate_data.py` | Script that simulates the data (fixed random seed, so it is reproducible) |

The data is **simulated**, not real AEMO data. It reproduces the main features of the market (evening price peaks, midday negative prices, heatwave spikes, wind lulls, gas-linked pricing, interconnector coupling), so figures should not be quoted as facts about the real market. A few missing values and duplicate rows were added on purpose for data-cleaning practice.

## What the notebook covers

1. Data quality checks and cleaning
2. Feature engineering (interval start, financial year, renewable share, residual demand)
3. Regional price comparison, including demand-weighted prices
4. Volatility: price duration curve and spike contribution
5. Daily patterns: evening peak, duck curve, negative prices
6. Seasonality heatmaps
7. Temperature-demand relationship and a peak-demand case study
8. Merit-order effect and wind-lull risk
9. Gas price link and regional price coupling
10. Battery arbitrage scenario model
11. Cap-contract hedging view
12. Demand forecasting with a Random Forest against a baseline
13. Key findings generated from the data

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Australian_Energy_Market_Analysis.ipynb
```

Change `FOCUS_REGION` in Step 1 to switch the deep-dive sections to another region. To regenerate the data, run `python generate_data.py`.

## Skills demonstrated

pandas data wrangling, time-series features, grouped analysis, risk metrics, correlation and regression, scenario modelling, scikit-learn forecasting with a time-based train/test split, and clear data visualisation, applied to Australian energy market concepts.
