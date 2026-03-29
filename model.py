import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from data_clean_categorical_columns import fitted_encoder, preprocess_categorical_data
from data_clean_numeric_columns import preprocess_numeric_data


def build_features(dataframe, encoder=None, fit_encoder=False, numeric_columns=None):
    numeric_data, numeric_columns = preprocess_numeric_data(
        dataframe, numeric_columns=numeric_columns
    )
    categorical_data, encoder = preprocess_categorical_data(
        dataframe, encoder=encoder, fit_encoder=fit_encoder
    )
    merged_data = pd.merge(numeric_data, categorical_data, on="Id")
    return merged_data, encoder, numeric_columns


train_data = pd.read_csv("train.csv", index_col="Id")
X, encoder, numeric_columns = build_features(
    train_data, encoder=fitted_encoder, fit_encoder=False
)
y = train_data["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_val)

print(f"Tuned model MAE: {mean_absolute_error(y_val, predictions)}")

model.fit(X, y)

test_data = pd.read_csv("test.csv", index_col="Id")
X_test, _, _ = build_features(
    test_data,
    encoder=encoder,
    fit_encoder=False,
    numeric_columns=numeric_columns,
)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

submission = pd.DataFrame(
    {
        "Id": X_test.index,
        "SalePrice": model.predict(X_test),
    }
)
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv")
