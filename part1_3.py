import pandas as pd
import statsmodels.api as sm


#import and clean the data
index_data = pd.read_excel("DATA/HW1_Hedge Fund.xlsx")
index_data["Date"] = pd.to_datetime(index_data["Date"])
factor_data = pd.read_excel("DATA/HW1_Factors.xlsx")
factor_data = factor_data.rename(columns={"Unnamed: 0": "Date"})
factor_data["Date"] = pd.to_datetime(factor_data["Date"], format='%Y%m')
for col in factor_data.columns:
    if col != 'Date':  
        factor_data[col] = factor_data[col] / 100



# distinguish up and down market for later use
up_mkt = factor_data[factor_data["Mkt-RF"]>0]
down_mkt = factor_data[factor_data["Mkt-RF"]<0]
print(up_mkt.info())



up_raw_data = pd.merge(index_data, up_mkt, how ="outer", on="Date")
hfri_columns = [col for col in up_raw_data.columns if 'HFRI' in col]
up_raw_data = up_raw_data.dropna(subset=hfri_columns)
factor_columns = ['RF', 'Mkt-RF']
up_raw_data = up_raw_data.dropna(subset=factor_columns)
# data of factor in 2025 May does not exist

down_raw_data = pd.merge(index_data, down_mkt, how ="outer", on="Date")
hfri_columns = [col for col in up_raw_data.columns if 'HFRI' in col]
down_raw_data = down_raw_data.dropna(subset=hfri_columns)
factor_columns = ['RF', 'Mkt-RF']
down_raw_data = down_raw_data.dropna(subset=factor_columns)
# data of factor in 2025 May does not exist



print(up_raw_data.info())
print(down_raw_data.info())
# check CAPM and simply do twice for different group


""" UP market CAPM
up_capm = pd.DataFrame()
up_capm[["Date","RF", "Mkt-RF"]]= up_raw_data[["Date","RF", "Mkt-RF"]]
for col in hfri_columns:
    # Excess return = HFRI return - Risk-free rate
    excess_return_col = f"up_{col}_excess"
    up_capm[excess_return_col] = up_raw_data[col] - up_raw_data['RF']

    
excess_columns = [col for col in up_capm.columns if 'excess' in col]

# regression
for i, col in enumerate(excess_columns):
    X = sm.add_constant(up_capm["Mkt-RF"])
    y = up_capm[col]
    model = sm.OLS(y, X).fit()
    print(model.summary())
"""


""" down market CAPM
down_capm = pd.DataFrame()
down_capm[["Date","RF", "Mkt-RF"]]= down_raw_data[["Date","RF", "Mkt-RF"]]
for col in hfri_columns:
    # Excess return = HFRI return - Risk-free rate
    excess_return_col = f"down_{col}_excess"
    down_capm[excess_return_col] = down_raw_data[col] - down_raw_data['RF']

    
excess_columns = [col for col in down_capm.columns if 'excess' in col]

# regression
for i, col in enumerate(excess_columns):
    X = sm.add_constant(down_capm["Mkt-RF"])
    y = down_capm[col]
    model = sm.OLS(y, X).fit()
    print(model.summary())
"""



def run_capm_summary(raw_df: pd.DataFrame, hfri_cols: list[str], tag: str):
    """
    回傳 CAPM 統計匯總表 (alpha, beta, t-stats, R²)
    
    raw_df: DataFrame，需包含 ['RF','Mkt-RF'] 以及 hfri_cols
    hfri_cols: 要跑迴歸的 hedge fund 欄位名稱
    tag: 'up' 或 'down'，會加在欄位名稱前面
    
    回傳: results_df (DataFrame)
    """
    results = []
    X = sm.add_constant(raw_df['Mkt-RF'])
    
    for col in hfri_cols:
        y = raw_df[col] - raw_df['RF']  # excess return
        model = sm.OLS(y, X).fit()
        
        results.append({
            'series': f"{tag}_{col}",
            'alpha': model.params['const'],
            'beta': model.params['Mkt-RF'],
            't_alpha': model.tvalues['const'],
            't_beta': model.tvalues['Mkt-RF'],
            'R2': model.rsquared
        })
    
    return pd.DataFrame(results)


up_results = run_capm_summary(up_raw_data, hfri_columns, tag="up")
down_results = run_capm_summary(down_raw_data, hfri_columns, tag="down")

print(up_results.to_string())
print(down_results.to_string())
