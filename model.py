from data_clean_numeric_columns import numeric_data,target
from data_clean_categorical_columns import merged_encoded_data, target
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# X = numeric_data
# y = target

merged_data = pd.merge(numeric_data,merged_encoded_data, on="Id")

X = merged_data
y = target

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

# Hyperparametrlar uchun qiymatlar
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

results = []
best_mae = float("inf")
best_params = None

# Har bir kombinatsiyani sinab koriw
from itertools import product

# for n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features in product(
#     param_grid["n_estimators"],
#     param_grid["max_depth"],
#     param_grid["min_samples_split"],
#     param_grid["min_samples_leaf"],
#     param_grid["max_features"]
# ):
#     model = RandomForestRegressor(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         max_features=max_features,
#         bootstrap=True,
#         random_state=42,
#         n_jobs=-1
#     )
    
#     model.fit(X_train, y_train)
#     prediction = model.predict(X_val)
#     mae = mean_absolute_error(y_val, prediction)
    
#     results.append({
#         "n_estimators": n_estimators,
#         "max_depth": max_depth,
#         "min_samples_split": min_samples_split,
#         "min_samples_leaf": min_samples_leaf,
#         "max_features": max_features,
#         "MAE": mae
#     })
    
#     if mae < best_mae:
#         best_mae = mae
#         best_params = {
#             "n_estimators": n_estimators,
#             "max_depth": max_depth,
#             "min_samples_split": min_samples_split,
#             "min_samples_leaf": min_samples_leaf,
#             "max_features": max_features
#         }

# Natijalarni DataFrame qib chiqariw
# results_df = pd.DataFrame(results)
# print(results_df)
# results_df.to_csv("tuning_results.csv")
tuned_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None
)
default_model =  RandomForestRegressor(random_state=1)

tuned_model.fit(X_train,y_train)
default_model.fit(X_train,y_train)

pred_t = tuned_model.predict(X_val)
pred_d = default_model.predict(X_val)

print(f"Tuned model MAE: {mean_absolute_error(y_val,pred_t)}")
print(f"Default model MAE: {mean_absolute_error(y_val,pred_d)}")