import pandas as pd


def preprocess_numeric_data(dataframe, numeric_columns=None):
    data = dataframe.copy()
    data["HasGarage"] = data["GarageYrBlt"].notna()
    data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

    if numeric_columns is None:
        numeric_columns = [
            column
            for column in data.columns
            if data[column].dtype not in ["string", "object"] and column != "SalePrice"
        ]

    return data[numeric_columns], numeric_columns


train_data = pd.read_csv("train.csv", index_col="Id")
numeric_data, numeric_columns = preprocess_numeric_data(train_data)
target = train_data["SalePrice"]
