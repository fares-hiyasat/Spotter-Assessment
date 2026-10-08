import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("train-test.csv")

print(df.shape)
print(df.info())
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
print(df["pickup"].value_counts().head(10))
print(df["delivery"].value_counts().head(10))
print(df["equipment"].value_counts())
print(df["pickup"].nunique())
print(df["delivery"].nunique())
print(df["equipment"].nunique())
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.dayofweek
print(df["day"])


X = df.drop(columns=["posted_rate", "load_id"])
y = df["posted_rate"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

main.fit(X_train, y_train)
y_pred = main.predict(X_val)

MAE = mean_absolute_error(y_val, y_pred)
RMSE = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(MAE, RMSE, r2)