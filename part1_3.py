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

# distinguish up and down market for later use
up_mkt = factor_data[factor_data["Mkt-RF"]>0]
down_mkt = factor_data[factor_data["Mkt-RF"]<0]


up_raw_data = pd.merge(index_data, factor_data, how ="outer", on="Date")
hfri_columns = [col for col in up_raw_data.columns if 'HFRI' in col]
up_raw_data = up_raw_data.dropna(subset=hfri_columns)
factor_columns = ['RF', 'Mkt-RF']
up_raw_data = up_raw_data.dropna(subset=factor_columns)
# data of factor in 2025 May does not exist

print(up_raw_data.info())


# check CAPM and simply do twice for different group