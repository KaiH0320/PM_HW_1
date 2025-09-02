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
#print(index_data.head(3))
#print(factor_data.head(3))

raw_data = pd.merge(index_data, factor_data, how ="outer", on="Date")
hfri_columns = [col for col in raw_data.columns if 'HFRI' in col]
raw_data = raw_data.dropna(subset=hfri_columns)
factor_columns = ['RF', 'Mkt-RF']
raw_data = raw_data.dropna(subset=factor_columns)
# data of factor in 2025 May does not exist

""" CAPM """
capm = pd.DataFrame()
capm[["Date","RF", "Mkt-RF"]]= raw_data[["Date","RF", "Mkt-RF"]]
for col in hfri_columns:
    # Excess return = HFRI return - Risk-free rate
    excess_return_col = f"{col}_excess"
    capm[excess_return_col] = raw_data[col] - raw_data['RF']

    
excess_columns = [col for col in capm.columns if 'excess' in col]

# regression
for i, col in enumerate(excess_columns):
    X = sm.add_constant(capm["Mkt-RF"])
    y = capm[col]
    model = sm.OLS(y, X).fit()
    print(model.summary())

# plotting
funds = excess_columns
n = len(funds)
rows, cols = (n + 2) // 3, 3   

fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

for ax, col in zip(axs.ravel(), funds):
    X = sm.add_constant(capm["Mkt-RF"])
    y = capm[col]
    model = sm.OLS(y, X, missing="drop").fit()
    
    # 畫散點
    ax.scatter(capm["Mkt-RF"], y, alpha=0.3, s=10)
    
    # 畫迴歸線
    xr = np.linspace(capm["Mkt-RF"].min(), capm["Mkt-RF"].max(), 200)
    yr = model.params[0] + model.params[1]*xr
    ax.plot(xr, yr, linestyle="--", linewidth=1.5)
    
    # 標題 + R²
    ax.set_title(col.replace("_excess",""))
    ax.text(0.02, 0.95, f"R²={model.rsquared:.2f}", transform=ax.transAxes, va="top")
    
    ax.grid(alpha=0.3)

# 移除多餘空白子圖
for i in range(len(funds), rows*cols):
    fig.delaxes(axs.ravel()[i])

fig.suptitle("CAPM Scatter Plots: Hedge Funds vs Market", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()

