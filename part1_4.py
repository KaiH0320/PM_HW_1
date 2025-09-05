import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

#import and clean the data
index_data = pd.read_excel("DATA/HW1_Hedge Fund.xlsx")
index_data["Date"] = pd.to_datetime(index_data["Date"])

print(index_data.info())

return_col = [col for col in index_data.columns if "HFRI" in col]
print(return_col)

results = []
for i, col in enumerate(return_col):
    y = index_data[col].rename("y")      
    y = y.dropna()                       

    model = AutoReg(y, lags=1, old_names=False).fit()

    ar1_key = "y.L1"
    results.append({
        "series": col,
        "AR1_coef": model.params[ar1_key],
        "t_stat":   model.tvalues[ar1_key],
        "p_value":  model.pvalues[ar1_key],
        "const":    model.params.get("const", float("nan"))
    })

df_ar1 = pd.DataFrame(results)
print(df_ar1)
