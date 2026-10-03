"""Predictive analytics: clean data -> compare models -> evaluate -> forecast next 12 months."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
sns.set_theme(style="whitegrid")
H = 12  # forecast / test horizon (months)

# ---------- 1. Load & clean ----------
raw = pd.read_csv("data/sales_history.csv", parse_dates=["Date"])
print(f"Raw rows: {len(raw)} | duplicates: {raw.duplicated('Date').sum()} | missing Sales: {raw.Sales.isna().sum()}")
df = raw.drop_duplicates("Date").set_index("Date").asfreq("MS")
# outliers: compare with rolling median, flag if > 2x or < 0.5x
med = df.Sales.rolling(7, center=True, min_periods=3).median()
out = (df.Sales > 2*med) | (df.Sales < 0.5*med)
print("Outliers flagged:", int(out.sum()), list(df.index[out].strftime("%Y-%m")))
df["Sales_clean"] = df.Sales.mask(out).interpolate(method="linear")
print("Missing after cleaning:", int(df.Sales_clean.isna().sum()))
y = df.Sales_clean

fig, ax = plt.subplots(figsize=(11,4))
ax.plot(df.Sales, color="lightgray", label="Raw (with issues)"); ax.plot(y, color="tab:blue", label="Cleaned")
ax.set(title="Step 1 - Data cleaning", ylabel="Monthly sales"); ax.legend(); plt.tight_layout()
plt.savefig("outputs/01_cleaning.png", dpi=130); plt.close()

# ---------- 2. Features ----------
def features(idx, hist):
    """calendar + trend + lag features. hist = series of known past values."""
    X = pd.DataFrame(index=idx)
    X["t"] = (idx.year-2019)*12 + idx.month
    for m in range(1,12): X[f"m{m}"] = (idx.month == m).astype(int)
    for L in (1,2,3,12): X[f"lag{L}"] = [hist.get(d - pd.DateOffset(months=L), np.nan) for d in idx]
    return X

train, test = y.iloc[:-H], y.iloc[-H:]
Xall = features(y.index, y)
cal = [c for c in Xall.columns if not c.startswith("lag")]
lagc = list(Xall.columns)

# ---------- 3. Models ----------
def seasonal_naive(tr, idx): return np.array([tr.get(d - pd.DateOffset(years=1)) for d in idx])
def fit_predict(model, cols, tr, idx, recursive):
    ok = Xall.loc[tr.index].dropna(subset=cols); model.fit(ok[cols], tr.loc[ok.index])
    if not recursive: return model.predict(features(idx, tr)[cols])
    hist = tr.copy(); preds = []
    for d in idx:                       # one step at a time, feeding predictions back as lags
        p = model.predict(features(pd.DatetimeIndex([d]), hist)[cols])[0]; hist.loc[d] = p; preds.append(p)
    return np.array(preds)

models = {
 "Seasonal Naive (baseline)": lambda tr, idx: seasonal_naive(tr, idx),
 "Linear Regression (trend+season)": lambda tr, idx: fit_predict(LinearRegression(), cal, tr, idx, False),
 "Random Forest (lags)": lambda tr, idx: fit_predict(RandomForestRegressor(300, random_state=42), lagc, tr, idx, True),
 "Gradient Boosting (lags)": lambda tr, idx: fit_predict(GradientBoostingRegressor(random_state=42), lagc, tr, idx, True),
}
def metrics(a, p):
    return dict(MAE=mean_absolute_error(a,p), RMSE=mean_squared_error(a,p)**.5, MAPE_pct=np.mean(np.abs((a-p)/a))*100)

# ---------- 4. Evaluate on the last 12 months (time-based split, no shuffling) ----------
preds, rows = {}, []
for n, f in models.items():
    preds[n] = f(train, test.index); rows.append({"Model": n, **metrics(test.values, preds[n])})
res = pd.DataFrame(rows).round(2).sort_values("MAPE_pct"); res.to_csv("outputs/model_comparison.csv", index=False)
print("\nHold-out results (last 12 months):\n", res.to_string(index=False))
best = res.iloc[0].Model; print("\nBest model:", best)

plt.figure(figsize=(11,4.5)); plt.plot(train.iloc[-30:], color="gray", label="Train (history)")
plt.plot(test, color="black", lw=2.5, label="Actual")
for n, p in preds.items(): plt.plot(test.index, p, "--", label=f"{n} ({res.set_index('Model').loc[n,'MAPE_pct']:.1f}% MAPE)")
plt.title("Step 2 - Model comparison on hold-out period"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("outputs/02_model_comparison.png", dpi=130); plt.close()

r = res.set_index("Model").MAPE_pct.sort_values(ascending=False)
plt.figure(figsize=(8,3.5)); sns.barplot(x=r.values, y=r.index, color="tab:blue"); plt.xlabel("MAPE % (lower is better)")
plt.title("Forecast error by model"); plt.tight_layout(); plt.savefig("outputs/03_error_by_model.png", dpi=130); plt.close()

# ---------- 5. Forecast the future with the best model (refit on all data) ----------
future = pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=H, freq="MS")
fc = pd.Series(models[best](y, future), index=future)
sigma = np.std(test.values - preds[best], ddof=1)
out_df = pd.DataFrame({"Forecast": fc.round(0), "Lower_95": (fc-1.96*sigma).round(0), "Upper_95": (fc+1.96*sigma).round(0)})
out_df.index.name = "Date"; out_df.to_csv("outputs/forecast_next_12_months.csv"); print("\n", out_df.to_string())

plt.figure(figsize=(11,4.5)); plt.plot(y, label="Historical (cleaned)"); plt.plot(fc, color="tab:red", lw=2, label=f"Forecast ({best})")
plt.fill_between(future, fc-1.96*sigma, fc+1.96*sigma, color="tab:red", alpha=.15, label="95% range")
plt.title("Step 3 - 12-month sales forecast"); plt.ylabel("Monthly sales"); plt.legend(); plt.tight_layout()
plt.savefig("outputs/04_forecast.png", dpi=130); plt.close()

# ---------- 6. Trend & seasonality insight ----------
dec = pd.DataFrame({"y": y}); dec["Year"] = dec.index.year; dec["Month"] = dec.index.month
yoy = dec.groupby("Year").y.sum(); print("\nYearly sales:\n", yoy.round(0).to_string())
fig, ax = plt.subplots(1,2, figsize=(11,4))
yoy.iloc[:-1].plot(kind="bar", ax=ax[0], color="tab:blue", title="Yearly sales (complete years)"); ax[0].tick_params(axis="x", rotation=0)
sea = (y / y.rolling(12, center=True, min_periods=12).mean()).groupby(y.index.month).mean()
sea.plot(kind="bar", ax=ax[1], color="tab:orange", title="Seasonality index (1.0 = average)"); ax[1].axhline(1, color="k", lw=.8)
plt.tight_layout(); plt.savefig("outputs/05_trend_seasonality.png", dpi=130); plt.close()
print("Seasonality peak month:", int(sea.idxmax()), "low month:", int(sea.idxmin()))
