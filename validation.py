import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv("train-test.csv")

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.dayofweek

X = df.drop(columns=["posted_rate", "load_id"])
y = df["posted_rate"]

numeric_features = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight", "market_index", "quote_signal", "month", "day"]
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
    ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

main.fit(X, y)


vl = pd.read_csv("validation.csv")

vl["date"] = pd.to_datetime(vl["date"])
vl["month"] = vl["date"].dt.month
vl["day"] = vl["date"].dt.dayofweek

validation_X = vl.drop(columns=["load_id"])

predictions = main.predict(validation_X)

output = pd.DataFrame({
    "load_id": vl["load_id"],
    "predicted_rate": predictions
})

output.to_csv("validation_predictions.csv", index=False)

