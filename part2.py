import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys


# 1) Load and clean data
unknown = pd.read_excel("/Users/huangyikai/Documents/FS SEM3/HW1/PM_HW_1/DATA/HW1_Unknown.xlsx")
unknown["Date"] = pd.to_datetime(unknown["Date"], format="%Y%m", errors="coerce")
unknown_cols = [c for c in unknown.columns if c != "Date"]


factors = pd.read_excel("/Users/huangyikai/Documents/FS SEM3/HW1/PM_HW_1/DATA/HW1_Factors.xlsx")
factors = factors.rename(columns={"Unnamed: 0": "Date"})
factors["Date"] = pd.to_datetime(factors["Date"], format="%Y%m", errors="coerce")
for col in factors.columns:
    if col != "Date":
        factors[col] = factors[col] / 100.0


# 2) Merge and compute excess returns
raw = pd.merge(unknown, factors, how="inner", on="Date")
# choose a representative Date as month end for readability
factor_cols = ["RF", "Mkt-RF", "SMB", "HML", "Mom"]

# Drop rows missing any unknown fund returns or required factors
raw = raw.dropna(subset=unknown_cols, how="any")
raw = raw.dropna(subset=factor_cols, how="any")

print(raw.info())


excess_df = pd.DataFrame({"Date": raw["Date"]})
for col in unknown_cols:
    excess_df[f"{col}_excess"] = raw[col] - raw["RF"]

# 3) Run 4-factor regressions for each unknown fund
results = []
coef_records = []

X = sm.add_constant(raw[["Mkt-RF", "SMB", "HML", "Mom"]])
for col in unknown_cols:
    y = raw[col] - raw["RF"]
    model = sm.OLS(y, X).fit()

    res = {
        "Fund": col,
        "alpha": model.params.get("const", np.nan),
        "beta_mkt": model.params.get("Mkt-RF", np.nan),
        "beta_smb": model.params.get("SMB", np.nan),
        "beta_hml": model.params.get("HML", np.nan),
        "beta_mom": model.params.get("Mom", np.nan),
        "t_alpha": model.tvalues.get("const", np.nan),
        "t_mkt": model.tvalues.get("Mkt-RF", np.nan),
        "t_smb": model.tvalues.get("SMB", np.nan),
        "t_hml": model.tvalues.get("HML", np.nan),
        "t_mom": model.tvalues.get("Mom", np.nan),
        "R2": model.rsquared,
        "AdjR2": model.rsquared_adj,
        "N": int(model.nobs)
    }
    results.append(res)

    coef_records.append({
        "Factor": "alpha",
        "Fund": col,
        "Loading": res["alpha"]
    })
    for fac_key, label in [("Mkt-RF", "Market"), ("SMB", "SMB"), ("HML", "HML"), ("Mom", "Momentum")]:
        coef_records.append({
            "Factor": label,
            "Fund": col,
            "Loading": model.params.get(fac_key, np.nan)
        })

summary_df = pd.DataFrame(results).set_index("Fund")
coef_df = pd.DataFrame(coef_records)

print("\nFour-Factor Regression Summary (monthly excess returns):")
print(summary_df.round(4).to_string())


# 4) Simple classification heuristics per fund
def classify_style(row: pd.Series) -> dict:
    beta = row["beta_mkt"]
    smb = row["beta_smb"]
    hml = row["beta_hml"]
    mom = row["beta_mom"]
    r2 = row["R2"]

    labels = []
    evidence = []

    # Equity vs. non-equity
    if abs(beta) >= 0.6 and r2 >= 0.5:
        labels.append("Equity-like")
        evidence.append(f"High market beta ({beta:.2f}) and R² {r2:.2f}")
    elif abs(beta) <= 0.3 and r2 <= 0.3:
        labels.append("Market-neutral/Alternative")
        evidence.append(f"Low beta ({beta:.2f}) and low R² {r2:.2f}")
    elif beta < -0.2:
        labels.append("Short/Defensive bias")
        evidence.append(f"Negative beta ({beta:.2f})")
    else:
        labels.append("Mixed/Moderate equity exposure")
        evidence.append(f"Moderate beta ({beta:.2f}) and R² {r2:.2f}")

    # Size tilt
    if smb >= 0.2:
        labels.append("Small-cap tilt")
        evidence.append(f"Positive SMB ({smb:.2f})")
    elif smb <= -0.2:
        labels.append("Large-cap tilt")
        evidence.append(f"Negative SMB ({smb:.2f})")

    # Value/Growth tilt
    if hml >= 0.2:
        labels.append("Value tilt")
        evidence.append(f"Positive HML ({hml:.2f})")
    elif hml <= -0.2:
        labels.append("Growth tilt")
        evidence.append(f"Negative HML ({hml:.2f})")

    # Momentum
    if mom >= 0.2:
        labels.append("Momentum tilt")
        evidence.append(f"Positive Momentum ({mom:.2f})")
    elif mom <= -0.2:
        labels.append("Contrarian/Anti-momentum")
        evidence.append(f"Negative Momentum ({mom:.2f})")

    # Alpha significance
    if abs(row["t_alpha"]) >= 2.0:
        labels.append("Non-zero alpha (statistically significant)")
        evidence.append(f"t(alpha)={row['t_alpha']:.2f}")

    return {
        "labels": ", ".join(labels),
        "evidence": "; ".join(evidence)
    }

class_rows = []
for fund, row in summary_df.iterrows():
    c = classify_style(row)
    class_rows.append({"Fund": fund, "Classification": c["labels"], "Evidence": c["evidence"]})

class_df = pd.DataFrame(class_rows).set_index("Fund")

print("\nStyle Classification:")
print(class_df.to_string())

# 5) Save outputs
out_dir = "/Users/huangyikai/Documents/FS SEM3/HW1/PM_HW_1"
summary_path = f"{out_dir}/part2_regression_summary.csv"
class_path = f"{out_dir}/part2_style_classification.csv"
summary_df.to_csv(summary_path)
class_df.to_csv(class_path)
print(f"\nSaved summary to: {summary_path}")
print(f"Saved classification to: {class_path}")

# 6) Plot factor loadings by fund
plt.figure(figsize=(10, 6))
pivot_coef = coef_df.pivot(index="Factor", columns="Fund", values="Loading").loc[["Market", "SMB", "HML", "Momentum"]]
pivot_coef.plot(kind="bar", figsize=(12, 6))
plt.title("Factor Loadings by Unknown Fund (FF4)")
plt.ylabel("Loading")
plt.xlabel("Factor")
plt.grid(alpha=0.3)
plt.tight_layout()
plot_path = f"{out_dir}/part2_factor_loadings.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved factor loadings plot to: {plot_path}")


