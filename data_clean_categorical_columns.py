import pandas as pd
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv("train.csv", index_col="Id")

OH_encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)

columns = data.columns
data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])

categorical = [i for i in columns if data[i].dtype in ["string",'object'] and not i in ["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","CentralAir","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]]
ordinals = data[["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","CentralAir","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]]
ordinals = ordinals.fillna("None")
ordinals["ExterQual"] = ordinals["ExterQual"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["ExterCond"] = ordinals["ExterCond"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["BsmtQual"] = ordinals["BsmtQual"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})

ordinals["BsmtCond"] = ordinals["BsmtCond"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["BsmtExposure"] = ordinals["BsmtExposure"].map({
    "None":0,
    "No":0,
    "Mn":1,
    "Av":2,
    "Gd":3
})
ordinals["BsmtFinType1"] = ordinals["BsmtFinType1"].map({
    "None":0,
    "Unf":1,
    "LwQ":2,
    "Rec":3,
    "BLQ":4,
    "ALQ":5,
    "GLQ":6
})
ordinals["BsmtFinType2"] = ordinals["BsmtFinType2"].map({
    "None":0,
    "Unf":1,
    "LwQ":2,
    "Rec":3,
    "BLQ":4,
    "ALQ":5,
    "GLQ":6
})
ordinals["HeatingQC"] = ordinals["HeatingQC"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["CentralAir"] = ordinals["CentralAir"].map({
    "N":0,
    "Y":1
})
ordinals['KitchenQual'] = ordinals["KitchenQual"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["FireplaceQu"] = ordinals["FireplaceQu"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["GarageQual"] = ordinals["GarageQual"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["GarageCond"] = ordinals["GarageCond"].map({
    "None":0,
    "Po":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
ordinals["PoolQC"] = ordinals["PoolQC"].map({
    "None":0,
    "Fa":1,
    "TA":2,
    "Gd":3,
    "Ex":4
})
data[categorical] = data[categorical].fillna("None")

categorical_data = data[categorical]
# encoded_categorical_data = pd.get_dummies(categorical_data)
OH_encoded_categorical_data = pd.DataFrame(OH_encoder.fit_transform(categorical_data), index=categorical_data.index,columns=OH_encoder.get_feature_names_out(categorical_data.columns))
target = data["SalePrice"]


# merged_encoded_data = pd.merge(encoded_categorical_data,ordinals,on="Id")
merged_encoded_data = pd.concat([OH_encoded_categorical_data, ordinals], axis=1)
