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


hf_data = pd.read_excel("DATA/HW1_Hedge Fund.xlsx")
hf_data["Date"] = pd.to_datetime(hf_data["Date"])

print(world_data.head())
print(hf_data.head())


raw_data = pd.merge(hf_data, world_data, how ="outer", on="Date")
hfri_columns = [col for col in raw_data.columns if 'HFRI' in col]
raw_data = raw_data.dropna(subset=hfri_columns)
raw_data = raw_data.dropna(subset=world_data.columns)


factor_data = pd.read_excel("DATA/HW1_Factors.xlsx")
factor_data = factor_data.rename(columns={"Unnamed: 0": "Date"})
factor_data["Date"] = pd.to_datetime(factor_data["Date"], format='%Y%m')
for col in factor_data.columns:
    if col != 'Date':  
        factor_data[col] = factor_data[col] / 100

rf_data = factor_data[["RF","Date"]] 

raw_data = pd.merge(raw_data, rf_data, how ="outer", on="Date")
raw_data = raw_data.dropna(subset=hfri_columns)
print(raw_data.info())

# regression 

results = []
for hf_col in hfri_columns:
    y = raw_data[hf_col] - raw_data["RF"]
    
    for country in world_columns: 
        X = sm.add_constant(raw_data[country] - raw_data["RF"])
        model = sm.OLS(y, X).fit()
        
        results.append({
            "HedgeFund": hf_col,
            "Country": country,
            "R2": model.rsquared
        })

df_r2 = pd.DataFrame(results)
pivot_r2 = df_r2.pivot(index="HedgeFund", columns="Country", values="R2")
print(pivot_r2.to_string())


# better visualization

def save_r2_heatmap(pivot_r2, filename="R2_heatmap.png"):
    data = pivot_r2.values
    n_rows, n_cols = data.shape

    # 動態設定圖大小
    fig_w = max(10, 0.7 * n_cols + 4)
    fig_h = max(6,  0.5 * n_rows + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 畫熱力圖
    im = ax.imshow(data, aspect="auto", cmap="BuGn")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("R²", rotation=90, va="center")

    # 標籤
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot_r2.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pivot_r2.index)

    # 在格子上加數字
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title("R² Heatmap: Hedge Funds vs. Country Markets")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {filename}")

save_r2_heatmap(pivot_r2, "R2_byCountry.png")
