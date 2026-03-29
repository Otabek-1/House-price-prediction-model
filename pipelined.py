import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("train.csv", index_col="Id")

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
numeric_cols = [col for col in X.columns if X[col].dtype != "object"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        n_jobs=-1,
    ))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_val)

print(mean_absolute_error(y_val, preds))