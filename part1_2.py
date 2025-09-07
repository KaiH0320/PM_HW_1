import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


#import and clean the data
index_data = pd.read_excel("DATA/HW1_Hedge Fund.xlsx")
index_data["Date"] = pd.to_datetime(index_data["Date"])
factor_data = pd.read_excel("DATA/HW1_Factors.xlsx")
factor_data = factor_data.rename(columns={"Unnamed: 0": "Date"})
factor_data["Date"] = pd.to_datetime(factor_data["Date"], format='%Y%m')
for col in factor_data.columns:
    if col != 'Date':  
        factor_data[col] = factor_data[col] / 100

print(factor_data.info)

#print(index_data.head(3))
#print(factor_data.head(3))

raw_data = pd.merge(index_data, factor_data, how ="outer", on="Date")
hfri_columns = [col for col in raw_data.columns if 'HFRI' in col]
raw_data = raw_data.dropna(subset=hfri_columns)
factor_columns = ['RF', 'Mkt-RF', 'SMB', 'HML', 'Mom']
raw_data = raw_data.dropna(subset=factor_columns)
# data of factor in 2025 May does not exist

""" FFM """
ffm = pd.DataFrame()
ffm[["Date","SMB","HML","Mom","RF", "Mkt-RF"]]= raw_data[["Date","SMB","HML","Mom","RF", "Mkt-RF"]]
for col in hfri_columns:
    # Excess return = HFRI return - Risk-free rate
    excess_return_col = f"{col}_excess"
    ffm[excess_return_col] = raw_data[col] - raw_data['RF']

    
excess_columns = [col for col in ffm.columns if 'excess' in col]

# regression
for i, col in enumerate(excess_columns):
    X = sm.add_constant(ffm[["Mkt-RF","SMB","HML","Mom"]])
    y = ffm[col]
    model = sm.OLS(y, X).fit()
    #print(model.summary())

# plotting
funds = excess_columns
n = len(funds)
rows, cols = (n + 2) // 3, 3   

fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

for ax, col in zip(axs.ravel(), funds):
    X = sm.add_constant(ffm[["Mkt-RF","SMB","HML","Mom"]])
    y = ffm[col]
    model = sm.OLS(y, X, missing="drop").fit()
    
    # scatter plot of actual vs predicted values
    y_pred = model.predict(X)
    ax.scatter(y, y_pred, alpha=0.3, s=10)
    
    # perfect prediction line (45-degree line)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    
    # title + R²
    ax.set_title(col.replace("_excess",""))
    ax.text(0.02, 0.95, f"R²={model.rsquared:.2f}", transform=ax.transAxes, va="top")
    
    # Add axis labels
    ax.set_xlabel('Actual Excess Return')
    ax.set_ylabel('Predicted Excess Return')
    
    ax.grid(alpha=0.3)


for i in range(len(funds), rows*cols):
    fig.delaxes(axs.ravel()[i])

fig.suptitle("FFM Model Fit: Actual vs Predicted Excess Returns", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()


def run_ff4_summary(raw_df: pd.DataFrame, series_cols: list[str]) -> pd.DataFrame:
    """
    Build FF4 (const + Mkt-RF + SMB + HML + Mom) regressions for given series.
    y = series - RF; X = [const, Mkt-RF, SMB, HML, Mom]
    Return a DataFrame with alpha, betas, t-stats, R², AdjR², N.
    """
    results = []
    X = sm.add_constant(raw_df[["Mkt-RF", "SMB", "HML", "Mom"]])
    for col in series_cols:
        y = raw_df[col] - raw_df["RF"]
        model = sm.OLS(y, X, missing="drop").fit()
        results.append({
            "series": col,
            "alpha": model.params.get("const", float("nan")),
            "beta_mkt": model.params.get("Mkt-RF", float("nan")),
            "beta_smb": model.params.get("SMB", float("nan")),
            "beta_hml": model.params.get("HML", float("nan")),
            "beta_mom": model.params.get("Mom", float("nan")),
            "t_alpha": model.tvalues.get("const", float("nan")),
            "t_mkt": model.tvalues.get("Mkt-RF", float("nan")),
            "t_smb": model.tvalues.get("SMB", float("nan")),
            "t_hml": model.tvalues.get("HML", float("nan")),
            "t_mom": model.tvalues.get("Mom", float("nan")),
            "R2": model.rsquared,
            "AdjR2": model.rsquared_adj,
            "N": int(model.nobs)
        })
    return pd.DataFrame(results)


# Generate and print FF4 summary table, also save to CSV
ff4_results = run_ff4_summary(raw_data, hfri_columns)
print(ff4_results.round(4).to_string())