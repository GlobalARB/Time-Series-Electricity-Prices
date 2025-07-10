import math
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from data import load_data


# Load real data (from data.py) or use synthetic for demo
df = load_data(
    from_month=1, to_month=12
)  # Assume this returns the merged DataFrame


# Sort values to ensure correct lag calculation
df.sort_values(by=["AreaCode", "DateTime"], inplace=True)


# Feature engineering
df["hour"] = df["DateTime"].dt.hour
df["dayofweek"] = df["DateTime"].dt.dayofweek
df["month"] = df["DateTime"].dt.month
df["net_load"] = df["TotalLoadValue"] - (
    df["Solar"] + df["Wind Onshore"] + df["Wind Offshore"]
)
df["lag1"] = df.groupby("AreaCode")["Price[Currency/MWh]"].shift(1)
df["lag24"] = df.groupby("AreaCode")["Price[Currency/MWh]"].shift(24)
df = pd.get_dummies(df, columns=["AreaCode", "ResolutionCode"], drop_first=True)
df.dropna(inplace=True)

features = [
    col for col in df.columns if col not in ["DateTime", "Price[Currency/MWh]"]
]
target = "Price[Currency/MWh]"


# Training and Evaluation with TS CV
tscv = TimeSeriesSplit(n_splits=5)
rmses = []

for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)

average_rmse = np.mean(rmses)
print(f"Average Gradient Boosting RMSE across folds: {average_rmse:.2f}")

