#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Country Momentum Long–Short (12–2) Portfolio
--------------------------------------------
- Signal: cumulative return over t-12..t-2 (product of (1+r) - 1), rebalanced monthly
- Portfolio: +50% equally across top 4 "winners", -50% equally across bottom 4 "losers"
- Self-financing (net = 0, gross = 100%)
- Diagnostics: annualized return/vol, Sharpe (IR), hit ratio, max drawdown, turnover
- Factor model: Carhart 4-factor with Newey–West (HAC) standard errors
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# === User-config paths ===
# =========================
WORLD_PATH  = Path("DATA/HW1_World.xlsx")
FACTOR_PATH = Path("DATA/HW1_Factors.xlsx")

# =========================
# === 0) Helper funcs   ===
# =========================
def annualize_return(monthly_series: pd.Series) -> float:
    return (1.0 + monthly_series.mean())**12 - 1.0

def annualize_vol(monthly_series: pd.Series) -> float:
    return monthly_series.std(ddof=1) * np.sqrt(12)

def sharpe_ir(monthly_series: pd.Series) -> float:
    # For self-financing LS, IR vs 0 (no RF subtraction)
    mu = monthly_series.mean()
    sig = monthly_series.std(ddof=1)
    return 0.0 if sig == 0 else (mu / sig) * np.sqrt(12)

def max_drawdown(nav: pd.Series) -> float:
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    return float(dd.min())

def tidy_summary(res) -> pd.DataFrame:
    return pd.DataFrame({
        "coef": res.params,
        "t_NW": res.tvalues,
        "p_NW": res.pvalues
    }).round({"coef":4, "t_NW":2, "p_NW":4})

# =======================================
# === 1) Load & prepare country data  ===
# =======================================
world_data = pd.read_excel(WORLD_PATH).rename(columns={"Unnamed: 0": "Date"})
world_data["Date"] = pd.to_datetime(world_data["Date"], format="%Y%m")

# Convert % to decimals for all country columns
for c in world_data.columns:
    if c != "Date":
        world_data[c] = world_data[c] / 100.0

countries = [c for c in world_data.columns if c != "Date"]
world_df = world_data.set_index("Date").sort_index()  # monthly returns by country

# =================================================
# === 2) Build 12–2 signal (cumulative t-12..t-2) ==
# =================================================
cr = pd.DataFrame(index=world_df.index)
for c in countries:
    # rolling window of 11 months (t-12..t-2 inclusive is 11 months),
    # then shift by 2 (one skip + one lag)
    cr[f"{c}_cumulative"] = (
        world_df[c]
        .rolling(window=11)
        .apply(lambda x: (1.0 + x).prod() - 1.0, raw=False)
        .shift(2)
    )

# keep rows where all signals needed for ranking exist
cr = cr.dropna(how="any")
cr_df = cr.copy()

# ================================================
# === 3) Long–Short weights & realized returns ===
# ================================================
ls_records = []
weights_by_date_ls = {}  # store monthly LS weights for turnover/costs

for dt, row in cr_df.iterrows():
    # 3.1) grab scores for current month dt
    scores = row.copy()
    # convert index "USA_cumulative" -> "USA"
    scores.index = [ix.replace("_cumulative", "") for ix in scores.index]
    scores = scores.loc[countries]  # ensure same country set/order

    # 3.2) pick winners/losers
    winners = scores.nlargest(4).index.tolist()
    losers  = scores.nsmallest(4).index.tolist()

    # 3.3) base weights: +0.50 equally on winners, -0.50 equally on losers
    w = pd.Series(0.0, index=countries, name="w_ls")
    if len(winners) > 0:
        w.loc[winners] = +0.50 / len(winners)
    if len(losers) > 0:
        w.loc[losers]  = -0.50 / len(losers)

    # 3.4) handle missing returns at month t by renormalizing each side separately
    ret_t = world_df.loc[dt, countries]
    valid = ret_t.notna() & w.ne(0.0)
    if valid.sum() != (len(winners) + len(losers)):
        # split sides
        wL = w[(w > 0) & valid]
        wS = w[(w < 0) & valid]

        if len(wL) > 0:
            w.loc[wL.index] = 0.50 * (wL / wL.abs().sum())
        else:
            w.loc[winners] = 0.0

        if len(wS) > 0:
            w.loc[wS.index] = -0.50 * (wS.abs() / wS.abs().sum())
        else:
            w.loc[losers] = 0.0

        # zero-out members with missing return this month
        w.loc[~valid] = 0.0

    # 3.5) realized LS return
    port_ret_ls = float((w * ret_t).sum())

    # store
    weights_by_date_ls[dt] = w
    ls_records.append({
        "Date": dt,
        "Winners": ", ".join(winners),
        "Losers":  ", ".join(losers),
        "LS_ret": port_ret_ls
    })

longshort_df = pd.DataFrame(ls_records).set_index("Date").sort_index()
ls = longshort_df["LS_ret"].astype(float)

# ==================================================
# === 4) Performance diagnostics (self-financing) ===
# ==================================================
ann_ret_ls = annualize_return(ls)
ann_vol_ls = annualize_vol(ls)
ir_ls      = sharpe_ir(ls)
hit_ratio  = float((ls > 0).mean())

cum_nav = (1.0 + ls).cumprod()  # Growth of $1
mdd = max_drawdown(cum_nav)

print("\n=== Long–Short Momentum Performance (12–2, top4 vs bottom4) ===")
print(f"Annualized return : {ann_ret_ls:.2%}")
print(f"Annualized vol    : {ann_vol_ls:.2%}")
print(f"Annualized Sharpe : {ir_ls:.2f}  (IR vs 0)")
print(f"Monthly hit ratio : {hit_ratio:.2%}")
print(f"Max drawdown      : {mdd:.2%}")

# ================================
# === 5) Turnover & costs (toy) ===
# ================================
turnover = []
prev_w = None
for dt in longshort_df.index:
    w = weights_by_date_ls[dt]
    if prev_w is not None:
        turnover.append(float((w - prev_w).abs().sum()))
    prev_w = w

avg_turnover = float(np.mean(turnover)) if turnover else 0.0
print(f"Avg monthly turnover (L1 weight change): {avg_turnover:.2%}")

# Simple cost model: cost_per_100pct = 40 bps per month (round-trip) as example
COST_PER_100PCT = 0.004
tc_per_month = COST_PER_100PCT * avg_turnover
ls_net = ls - tc_per_month

ann_ret_ls_net = annualize_return(ls_net)
ir_ls_net      = sharpe_ir(ls_net)
print(f"Net annualized return (toy costs) : {ann_ret_ls_net:.2%}")
print(f"Net annualized Sharpe (toy costs) : {ir_ls_net:.2f}")

# =========================================
# === 6) Carhart (Newey–West HAC) OLS   ===
# =========================================
factors = pd.read_excel(FACTOR_PATH).rename(columns={"Unnamed: 0": "Date"})
factors["Date"] = pd.to_datetime(factors["Date"], format="%Y%m")

# Convert % to decimals for factor columns
for c in factors.columns:
    if c != "Date":
        factors[c] = factors[c] / 100.0

reg_df = (
    longshort_df[["LS_ret"]]
    .join(factors.set_index("Date")[["RF","Mkt-RF","SMB","HML","Mom"]], how="inner")
    .dropna()
)

y = reg_df["LS_ret"]  # self-financing portfolio -> use raw LS return
X = sm.add_constant(reg_df[["Mkt-RF","SMB","HML","Mom"]])

ols_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
print("\n=== Carhart OLS with Newey–West (HAC, 12 lags) ===")
print(ols_hac.summary())

tidy = tidy_summary(ols_hac)
print("\nTidy (HAC):")
print(tidy)
print(f"\nR^2: {ols_hac.rsquared:.3f} | Adj R^2: {ols_hac.rsquared_adj:.3f}")

alpha_m  = float(ols_hac.params["const"])
alpha_t  = float(ols_hac.tvalues["const"])
alpha_p  = float(ols_hac.pvalues["const"])
mom_b    = float(ols_hac.params["Mom"])
mom_t    = float(ols_hac.tvalues["Mom"])
mom_p    = float(ols_hac.pvalues["Mom"])

print("\nOne-liner takeaway (LS):")
print(f"- Alpha (monthly): {alpha_m:.4%}, t={alpha_t:.2f}, p={alpha_p:.4f}")
print(f"- MOM loading    : {mom_b:.2f}, t={mom_t:.2f}, p={mom_p:.4f}")

# =======================
# === 7) Simple plots ===
# =======================
plt.figure(figsize=(9,4))
plt.plot(cum_nav.index, cum_nav.values)
plt.title("Momentum Long–Short (12–2) – Growth of $1")
plt.ylabel("NAV")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Drawdown path
dd = cum_nav / cum_nav.cummax() - 1.0
plt.figure(figsize=(9,3.5))
plt.plot(dd.index, dd.values)
plt.title("Drawdown – Momentum Long–Short")
plt.ylabel("Drawdown")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================
# === 8) Save key outputs ===
# ===========================
out_dir = WORLD_PATH.parent
(longshort_df.assign(NAV=cum_nav)).to_csv(out_dir / "longshort_momentum_results.csv")
tidy.to_csv(out_dir / "longshort_carhart_tidy.csv")

print(f"\nSaved results to:\n- {out_dir / 'longshort_momentum_results.csv'}\n- {out_dir / 'longshort_carhart_tidy.csv'}")
