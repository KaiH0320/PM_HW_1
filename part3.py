from curses import window
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


# read and clean the data
world_data = pd.read_excel('/Users/huangyikai/Documents/FS SEM3/HW1/PM_HW_1/DATA/HW1_World.xlsx')
world_data = world_data.rename(columns={"Unnamed: 0": "Date"})
world_data["Date"] = pd.to_datetime(world_data["Date"], format='%Y%m')
for col in world_data.columns:
    if col != 'Date':  
        world_data[col] = world_data[col] / 100
world_columns = [col for col in world_data.columns if "Date" not in col]

print(world_data.info())

cr = pd.DataFrame()
cr["Date"] = world_data["Date"]
# interesting pandas function prod will return prod of each row you assign
for country in world_columns:
    returns = world_data[country]
    cumulative_return = returns.rolling(window=11).apply(
        lambda x:(1+x).prod() -1, raw=False
    ).shift(2) # Shift by 2: 1 month skip + 1 month lag

    cr[f"{country}_cumulative"] = cumulative_return
cr_columns = [col for col in cr.columns if "Date" not in col]
cr = cr.dropna(subset=cr_columns)
print(cr.info())


# 假設已有：
world_df = world_data.set_index("Date")     # 各國月報酬（小數），列=月份
cr_df    = cr.set_index("Date")             # 月 t 用來排序的 11M 累積報酬
countries = [c for c in world_df.columns]     # 20 個國家欄位

records = []
# 使用 t-1 的排序分數在 t 月建倉
score_cols = [c for c in cr_df.columns if c.endswith("_cumulative")]
lagged_scores = cr_df[score_cols].shift(1)

for dt in lagged_scores.index:
    prev = lagged_scores.loc[dt].dropna()
    if prev.empty:
        continue

    # 還原成國家名，並僅保留 20 國
    prev.index = [c.replace("_cumulative", "") for c in prev.index]
    prev = prev.loc[[c for c in countries if c in prev.index]]
    if prev.empty:
        continue

    winners = prev.nlargest(4).index.tolist()
    losers  = prev.nsmallest(4).index.tolist()
    middle  = [c for c in countries if c not in winners + losers]

    # 建立權重（遇缺值將於下步再正規化）
    w = pd.Series(0.0, index=countries, name="weight")
    if len(winners) == 0 or len(middle) == 0:
        continue
    w.loc[winners] = 0.50 / len(winners)
    w.loc[middle]  = 0.50 / len(middle)

    # 用 t 月報酬計算投組報酬；若有缺值，對非零權重重新正規化
    if dt not in world_df.index:
        continue
    ret_t = world_df.loc[dt, countries].astype(float)
    mask = ret_t.notna() & w.ne(0)
    if mask.any():
        w_adj = w.where(mask, 0.0)
        if w_adj.sum() != 0:
            w_adj = w_adj / w_adj.sum()
        port_ret = (w_adj * ret_t).sum()
    else:
        port_ret = np.nan

    records.append({
        "Date": dt,
        "Winners": ", ".join(winners),
        "Losers":  ", ".join(losers),
        "Middle12": ", ".join(middle),
        "LongOnly_ret": port_ret
    })

longonly_df = pd.DataFrame(records).set_index("Date").sort_index()
print(longonly_df.head())


# === 1) 載入四因子並與投組對齊 ===
factor_data = pd.read_excel("DATA/HW1_Factors.xlsx").rename(columns={"Unnamed: 0": "Date"})
factor_data["Date"] = pd.to_datetime(factor_data["Date"], format="%Y%m")

# 因子檔通常是百分比；轉小數
for c in factor_data.columns:
    if c != "Date":
        factor_data[c] = factor_data[c] / 100.0

df = (
    longonly_df[["LongOnly_ret"]]
      .join(factor_data.set_index("Date")[["RF","Mkt-RF","SMB","HML","Mom"]], how="inner")
      .dropna()
)

# === 2) 績效指標（年化） ===
port   = df["LongOnly_ret"]           # 月報酬（小數）
excess = port - df["RF"]              # 月超額報酬
# here we car about the expect value, not as a investoer care about the "real return" 
AF = 12  # 月資料 -> 年化因子

# 1. 年化報酬
# (a) 算術年化（學術檢定用，和 t-test 搭配）
ann_ret_arith = port.mean() * AF

# (b) 幾何年化（投資人角度，對數法更嚴謹）
ann_ret_geo = np.exp(AF * np.log1p(port).mean()) - 1


# 2. 年化波動
ann_vol = port.std(ddof=1) * np.sqrt(AF)

# 3. 年化 Sharpe ratio
mean_excess_ann = excess.mean() * AF
vol_excess_ann  = excess.std(ddof=1) * np.sqrt(AF)
ann_sharpe = mean_excess_ann / vol_excess_ann

print("\n=== Long-Only Momentum Performance ===")
print(f"Annualized return (Arithmetic)  : {ann_ret_arith:.2%}")
print(f"Annualized return (Geometric)   : {ann_ret_geo:.2%}")
print(f"Annualized vol                  : {ann_vol:.2%}")
print(f"Annualized Sharpe (=(12*mean(R-RF))/(sqrt(12)*sd(R-RF))) : {ann_sharpe:.2f}")
print(f"Mean excess (annualized)        : {mean_excess_ann:.2%}")
print(f"Vol excess  (annualized)        : {vol_excess_ann:.2%}")

# === 3) 累積報酬圖 ===
cum = (1 + port).cumprod() - 1
plt.figure(figsize=(8,4))
plt.plot(cum.index, cum.values)
plt.title("Momentum Long-Only Portfolio – Cumulative Return")
plt.ylabel("Cumulative Return")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# === 4) 四因子回歸（Carhart） on EXCESS returns ===
y = excess
X = sm.add_constant(df[["Mkt-RF","SMB","HML","Mom"]])

# 一般 OLS
ols = sm.OLS(y, X).fit()

print("\n=== OLS (classical SE) ===")
print(ols.summary())


# 精簡輸出（OLS）
tidy = pd.DataFrame({
    "coef": ols.params.round(4),
    "t":    ols.tvalues.round(2),
    "p":    ols.pvalues.round(4)
})
print("\nTidy four-factor regression (OLS):\n", tidy)
print(f"\nR^2: {ols.rsquared:.3f} | Adj R^2: {ols.rsquared_adj:.3f}")

# === 5) 一句話結論模板（看 alpha 與 MOM） ===
alpha_coef = ols.params["const"]; alpha_t = ols.tvalues["const"]; alpha_p = ols.pvalues["const"]
mom_coef   = ols.params["Mom"];   mom_t   = ols.tvalues["Mom"];   mom_p   = ols.pvalues["Mom"]

print("\nSummary note:")
print(f"- Alpha (monthly): {alpha_coef:.4%}, t={alpha_t:.2f}, p={alpha_p:.4f}")
print(f"- MOM loading    : {mom_coef:.2f}, t={mom_t:.2f}, p={mom_p:.4f}")
print("Interpretation: If alpha is insignificant and MOM is significantly > 0, "
      "the momentum factor largely explains the strategy’s profit.")

