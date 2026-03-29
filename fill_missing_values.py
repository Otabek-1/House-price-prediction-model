import pandas as pd

df = pd.DataFrame({
    "gender": ["male", "female", "female", "male"],
    "city": ["Tashkent", "Samarkand", "Bukhara", "Tashkent"],
    "level": ["low", "high", "medium", "low"]
})

# gender: binary
df["gender"] = df["gender"].map({
    "male": 0,
    "female": 1
})

# level: tartibli
df["level"] = df["level"].map({
    "low": 0,
    "medium": 1,
    "high": 2
})

# city: one-hot encoding
df = pd.get_dummies(df, columns=["city"])

print(df)