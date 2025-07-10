import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import math
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


# LSTM Model
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx : idx + self.seq_length, :-1],
            self.data[idx + self.seq_length, -1],
        )


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# Training and Evaluation with TS CV
tscv = TimeSeriesSplit(n_splits=5)
rmses = []

for train_idx, test_idx in tscv.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features + [target]])
    test_scaled = scaler.transform(test_df[features + [target]])

    seq_length = 24  # Past day
    train_dataset = TimeSeriesDataset(train_scaled, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LSTMModel(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(20):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq.float())
            loss = criterion(y_pred, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

    # Test
    test_inputs = np.array(
        [
            test_scaled[i : i + seq_length, :-1]
            for i in range(len(test_scaled) - seq_length)
        ]
    )
    test_inputs = torch.tensor(test_inputs).float()
    model.eval()
    with torch.no_grad():
        lstm_pred_scaled = model(test_inputs).numpy()

    dummy = np.hstack((test_scaled[seq_length:, :-1], lstm_pred_scaled))
    lstm_pred = scaler.inverse_transform(dummy)[:, -1]
    lstm_actual = test_df[target].values[seq_length:]
    rmse = math.sqrt(mean_squared_error(lstm_actual, lstm_pred))
    rmses.append(rmse)

average_rmse = np.mean(rmses)
print(f"Average LSTM RMSE across folds: {average_rmse:.2f}")

# For baseline comparison (add similar CV)
# ... (implement baseline in loop and compare)
