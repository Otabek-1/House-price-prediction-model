import pandas as pd
from sklearn.preprocessing import OneHotEncoder


ORDINAL_COLUMNS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]

ORDINAL_MAPPINGS = {
    "ExterQual": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "ExterCond": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtQual": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtCond": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtExposure": {"None": 0, "No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "HeatingQC": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "CentralAir": {"N": 0, "Y": 1},
    "KitchenQual": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "FireplaceQu": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "GarageQual": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "GarageCond": {"None": 0, "Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "PoolQC": {"None": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
}


def preprocess_categorical_data(dataframe, encoder=None, fit_encoder=False):
    data = dataframe.copy()
    data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])

    categorical_columns = [
        column
        for column in data.columns
        if data[column].dtype in ["string", "object"] and column not in ORDINAL_COLUMNS
    ]

    ordinals = data[ORDINAL_COLUMNS].copy().fillna("None")
    for column, mapping in ORDINAL_MAPPINGS.items():
        ordinals[column] = ordinals[column].map(mapping)

    data[categorical_columns] = data[categorical_columns].fillna("None")
    categorical_data = data[categorical_columns]

    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_values = encoder.fit_transform(categorical_data)
    else:
        encoded_values = encoder.transform(categorical_data)

    encoded_categorical_data = pd.DataFrame(
        encoded_values,
        index=categorical_data.index,
        columns=encoder.get_feature_names_out(categorical_data.columns),
    )

    merged_encoded_data = pd.concat([encoded_categorical_data, ordinals], axis=1)
    return merged_encoded_data, encoder


train_data = pd.read_csv("train.csv", index_col="Id")
merged_encoded_data, fitted_encoder = preprocess_categorical_data(train_data, fit_encoder=True)
target = train_data["SalePrice"]
