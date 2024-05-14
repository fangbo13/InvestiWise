import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def fetch_data(stock_code, training_years):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=training_years)
    return yf.download(stock_code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

def feature_engineering(data, prediction_days):
    data['Returns'] = data['Close'].pct_change()
    data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
    data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['slowk'], data['slowd'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    upperband, middleband, lowerband = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['upperband'] = upperband
    data['middleband'] = middleband
    data['lowerband'] = lowerband
    data['MOM'] = talib.MOM(data['Close'], timeperiod=10)

    data['Future_Close'] = data['Close'].shift(-prediction_days)
    data['Target'] = (data['Future_Close'] > data['Close']).astype(int)

    return data.dropna()

def dataset_generation_for_LSTM(x, pred, Impact_length=30, Split_ratio=0.7):
    num_features = x.shape[1]
    pred_array = pred.values.reshape(-1, 1)
    data_length = len(pred_array)
    selected_data = x
    count_start = len(selected_data) - data_length
    temp = []

    for i in range(data_length):
        start_index = count_start - Impact_length
        end_index = count_start
        temp.append(selected_data[start_index:end_index].tolist())
        count_start += 1

    max_length = max(len(seq) for seq in temp)
    padded_temp = [seq + [[0] * num_features] * (max_length - len(seq)) for seq in temp]
    Features = np.array(padded_temp)
    Label = np.array(pred_array)

    x_train, x_test, y_train, y_test = train_test_split(
        Features, Label, test_size=1 - Split_ratio, random_state=500)

    return x_train, x_test, y_train, y_test, Features

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def train_model(stock_code, training_years, model_type='LSTM', prediction_days=5, hidden_dim=32, num_epochs=100):
    data = fetch_data(stock_code, training_years)
    data = feature_engineering(data, prediction_days)
    x = data[[col for col in data.columns if col != 'Target']]
    y = data['Target']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test, alldata = dataset_generation_for_LSTM(x, y, Impact_length=30)
    
    if model_type == 'LSTM':
        model = LSTMModel(input_dim=x_train.shape[2], hidden_dim=hidden_dim, num_layers=2, output_dim=1, dropout_prob=0.2)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        test_data = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        best_auc = 0.0
        patience = 10
        trigger_times = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            y_test_pred = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)
                    y_test_pred.extend(y_pred.cpu().numpy().flatten())

            y_test_pred = np.array(y_test_pred)
            predictions = (y_test_pred > 0.5).astype(int)
            report = classification_report(y_test, predictions, zero_division=1)
            fpr, tpr, _ = roc_curve(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_pred)

            if roc_auc > best_auc:
                best_auc = roc_auc
                trigger_times = 0
            else:
                trigger_times += 1

            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}, AUC: {roc_auc}")

        return {
            "predictions": predictions.tolist(),
            "prediction_proba": y_test_pred.tolist(),
            "classification_report": report,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "roc_auc": best_auc
        }
    else:
        raise ValueError("Unsupported model type")

# Example usage
if __name__ == "__main__":
    results = train_model("AAPL", 5, 'LSTM')
    print(results)
