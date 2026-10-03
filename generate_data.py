"""Create a synthetic monthly sales history (no dataset was provided with the task).
Includes realistic problems to clean: missing values, outliers, duplicate row."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
dates = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
t = np.arange(len(dates))
season = 1 + 0.18*np.sin(2*np.pi*(dates.month-4)/12) + np.where(dates.month.isin([11,12]), 0.22, 0)
sales = np.asarray((50000 + 650*t) * season * rng.normal(1, 0.04, len(t)), dtype=float)
sales[np.asarray((dates >= "2020-03-01") & (dates <= "2020-06-01"))] *= 0.7      # a dip (shock)
df = pd.DataFrame({"Date": dates, "Sales": sales.round(2)})
df["Marketing_Spend"] = (df.Sales*0.08*rng.normal(1,.1,len(df))).round(2)
df.loc[rng.choice(len(df), 4, replace=False), "Sales"] = np.nan        # missing
df.loc[[30, 61], "Sales"] *= 3.2                                         # outliers
df = pd.concat([df, df.iloc[[10]]]).sort_values("Date").reset_index(drop=True)  # duplicate
df.to_csv("data/sales_history.csv", index=False); print(df.shape)
