# Predictive Analytics Using Historical Data

Forecasting monthly sales for the next 12 months using regression and machine-learning models in Python (scikit-learn), with data cleaning, a fair time-based evaluation and visualised predictions.

> **Data note:** no dataset was provided with the task, so `generate_data.py` creates a synthetic monthly sales history (Jan 2019 - Sep 2026) that includes real-world problems: missing values, outliers, a duplicate row and a 2020 demand shock. To use real data, keep the columns `Date, Sales` and re-run `forecast.py`.

## Project structure
| File | Purpose |
|------|---------|
| `generate_data.py` | Creates the synthetic dataset |
| `forecast.py` | Cleaning, feature engineering, model comparison, evaluation, forecast, charts |
| `data/sales_history.csv` | Raw input data (94 rows) |
| `outputs/` | Charts, `model_comparison.csv`, `forecast_next_12_months.csv` |

## How to run
```bash
pip install -r requirements.txt
python generate_data.py     # optional, data is already included
python forecast.py
```

## Method
1. **Cleaning:** removed 1 duplicate month, flagged 2 outliers (values more than 2x or less than 0.5x the rolling median) and linearly interpolated them with the 4 missing values. Result: a complete monthly series with no gaps.
2. **Features:** trend index, month-of-year dummies (seasonality) and lag features (1, 2, 3 and 12 months back).
3. **Models compared:** Seasonal Naive (baseline), Linear Regression (trend + seasonality), Random Forest and Gradient Boosting (lag features, forecasted recursively).
4. **Evaluation:** time-based split, no shuffling. Models train on history and are tested on the **last 12 months** using MAE, RMSE and MAPE.
5. **Forecast:** the best model is refitted on all data and predicts the next 12 months, with an approximate 95% range from the hold-out error.

## Results (last 12 months hold-out)
| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **Linear Regression (trend + season)** | **4,574** | **6,129** | **4.6%** |
| Random Forest (lags) | 5,700 | 6,616 | 5.1% |
| Gradient Boosting (lags) | 5,751 | 6,735 | 5.1% |
| Seasonal Naive (baseline) | 8,945 | 9,891 | 7.9% |

The simple linear model wins: the series has a clear steady trend plus stable seasonality, which tree models can't extrapolate as well. Every model beats the naive baseline.

## Business insights
- **Sales grow steadily:** yearly revenue rose from about 0.67M (2019, 2020) to 1.26M (2025), with 2025 up about 9.5% on 2024.
- **Next 12 months (Oct 2026 - Sep 2027):** forecast of about 1.44M, roughly **8% above** the last 12 months.
- **Seasonality:** strong summer peak and a Nov-Dec holiday bump, with the low point in January. Plan inventory and staffing ahead of both peaks.
- **2020 shock:** the dip was temporary and growth resumed, so the model treats it as noise rather than a trend change.
- **Caution:** forecasts assume trend and seasonality continue. Accuracy is about 95% on average (MAPE 4.6%) and the shaded 95% range shows the uncertainty.

## Charts
`01_cleaning` · `02_model_comparison` · `03_error_by_model` · `04_forecast` · `05_trend_seasonality` (all in `outputs/`).

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Learning outcomes
Data cleaning, feature engineering for time series, regression and ensemble models, time-based validation, forecast accuracy metrics (MAE, RMSE, MAPE) and data-driven forecasting.
