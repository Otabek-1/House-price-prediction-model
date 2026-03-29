## EXPERIMENT 1:

MAE: 24442.94
MODEL: DecisionTreeRegressor

FEATURES:
- Only numeric features

DATA SPLIT:
<!-- - test_size: 0.2
- random_state: 42 -->
- No data split.


HYPERPARAMETERS:
- default

TUNING:
- No

NOTES:
- Baseline model
- No categorical features
- Next: try RandomForest or feature engineering

## EXPERIMENT 2:
MAE: 17359.532821917808 MODEL: RandomForestRegressor

NOTES:
- Same with experiment 1
- Next: try out other models.

## EXPERIMENT 3:
MAE: 28649.923835616435 MODEL: KNN

NOTES:
- Worst model used

## EXPERIMENT 4:
MAE: 17496.664730593606 MODEL: RandomForestRegressor

NOTES:
- Filled missing values of numeric columns but result is not expected.

## EXPERIMENT 5:
MAE: 17518.71866666667 MODEL: RandomForestRegressor

NOTES:
- Added boolen <code>HasGarage</code> feature and filled with <code>GarageYrBlt</code>'s values.
- Good news!

## EXPERIMENT 6:
MAE: 17447.210977168954 MODEL: RandomForestRegressor
NOTES:
- Removed filler values from <code>LotFrontage</code> column.
- NaN = signal.

## EXPERIMENT 7:
MAE: 17100.23315068493 MODEL: RandomForestRegressor
NOTES:
- Cleaned categorical datas with simple OH encoding.
- merged all cleaned datas.
- Good!
- Next: Ordinal encoding.

## EXPERIMENT 8:
MAE: 16680.276356164384
- Ordinal encoding used with most stupid way.

## EXPERIMENT 9:
MAE: 16379.352240
- Added Hyperparametrs to model and tested with SimpleImputer (but SimpleImputer didn't work in this model so it has removed).