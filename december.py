import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv("train-test.csv")

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.dayofweek

X = df[[
    "pickup", "delivery", "distance", "equipment",
    "weight", "month", "day"
]]
y = df["posted_rate"]

numeric_features = ["distance", "weight", "month", "day"]
categorical_features = ["pickup", "delivery", "equipment"]

num_pl = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pl = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

pre_p = ColumnTransformer([
    ("num", num_pl, numeric_features),
    ("cat", cat_pl, categorical_features)
])

main = Pipeline([
    ("preprocessor", pre_p),
    ("model", GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

main.fit(X, y)


dec = pd.read_csv("december-chart-inputs.csv")
dec["date"] = pd.to_datetime(dec["date"])
dec["month"] = dec["date"].dt.month
dec["day"] = dec["date"].dt.dayofweek

dec_X = dec[[
    "pickup", "delivery", "distance",
    "equipment", "weight", "month", "day"
]]

dec["predicted_rate"] = main.predict(dec_X)

dec[[
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate"
]].to_csv("december_predictions.csv", index=False)
print(dec[["date", "predicted_rate"]])