import pandas as pd

data = pd.read_csv("train.csv",index_col="Id")
data["HasGarage"] = data["GarageYrBlt"].notna()
cols = data.columns

numeric = [i for i in cols if not data[i].dtype in ["string","object"] and not i == "SalePrice"]
# data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].mean())
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
# print(data[numeric].isna().sum())
numeric_data = data[numeric]
target = data["SalePrice"]