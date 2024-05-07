import numpy as np
import pandas as pd
import talib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def fetch_data(stock_code, training_years):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=training_years)
    return yf.download(stock_code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

def feature_engineering(data):
    data['Returns'] = data['Close'].pct_change()
    data['RSI'] = talib.RSI(data['Close'])
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'])
    # Adding more technical indicators
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)  # Simple Moving Average
    data['EMA'] = talib.EMA(data['Close'], timeperiod=20)  # Exponential Moving Average
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)  # Average True Range
    return data.dropna()

def prepare_data(data):
    data = feature_engineering(data)
    X = data[['RSI', 'MACD', 'MACD_signal', 'SMA', 'EMA', 'ATR']]
    y = (data['Returns'] > 0).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(stock_code, training_years, model_type='RF'):
    data = fetch_data(stock_code, training_years)
    X_train, X_test, y_train, y_test = prepare_data(data)

    if model_type == 'RF':
        model = RandomForestClassifier(n_estimators=100)
        parameters = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
        clf = GridSearchCV(model, parameters, cv=3)
    elif model_type == 'SVM':
        model = SVC(probability=True)
        parameters = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 'scale']}
        clf = GridSearchCV(model, parameters, cv=3)
    else:
        raise ValueError("Unsupported model type")

    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    predictions = best_model.predict(X_test)
    report = classification_report(y_test, predictions)
    probs = best_model.predict_proba(X_test)[:, 1] if model_type == 'RF' else best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)

    return {
        "best_params": clf.best_params_,
        "classification_report": report,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "roc_auc": roc_auc
    }

# Example usage
if __name__ == "__main__":
    results = train_model("AAPL", 5, 'RF')
    print(results)
