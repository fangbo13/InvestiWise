import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def get_trained_lstm_model(stock_code='AAPL', start_date='2010-01-01', end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # 下载数据
    df = yf.download(stock_code, start=start_date, end=end_date)
    data = df['Close'].values.reshape(-1, 1)
    
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 创建训练和测试数据
    sequence_length = 60
    train_size = int(len(data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]

    x_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i-sequence_length:i, 0])
        y_train.append(train_data[i, 0])

    x_test, y_test = [], []
    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i-sequence_length:i, 0])
        y_test.append(test_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 训练模型
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
    model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stop])
    
    return model, scaler, df, x_train, y_train, x_test, y_test

def predict_future_prices(stock_code='AAPL', days_to_predict=7):
    model, scaler, df, x_train, y_train, x_test, y_test = get_trained_lstm_model(stock_code=stock_code)
    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)
    
    # 预测测试集
    test_predictions = model.predict(x_test)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 使用您提供的方法预测未来N天
    sequence_length = 60
    x_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    future_predictions = []
    for _ in range(days_to_predict):
        predicted_price = model.predict(x_input)
        future_predictions.append(predicted_price[0, 0])
        
        # 将 predicted_price 重新形状为 (1, 1, 1) 以匹配 x_input 的形状
        predicted_price_reshaped = np.reshape(predicted_price, (1, 1, 1))
        x_input = np.append(x_input[:, 1:, :], predicted_price_reshaped, axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    historical_data = df['Close'].tolist()
    dates = df.index.strftime('%Y-%m-%d').tolist()
    prediction_dates = pd.date_range(start=dates[-1], periods=days_to_predict + 1, inclusive='right').strftime('%Y-%m-%d').tolist()
    
    return {
        'historical': historical_data,
        'dates': dates,
        'test': y_test.flatten().tolist(),
        'test_predictions': test_predictions.flatten().tolist(),
        'predictions': future_predictions.tolist(),
        'prediction_dates': prediction_dates
    }
