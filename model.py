from data_clean_numeric_columns import numeric_data, target
from data_clean_categorical_columns import merged_encoded_data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


merged_data = pd.merge(numeric_data, merged_encoded_data, on="Id")

X = merged_data
y = target

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
