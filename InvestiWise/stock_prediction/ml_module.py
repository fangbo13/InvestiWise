import numpy as np
import pandas as pd
import talib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fetch_data(stock_code, training_years):
    """Fetch historical stock data from Yahoo Finance."""
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=training_years)
    return yf.download(stock_code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

def feature_engineering(data):
    """Generate technical indicators to use as features for machine learning."""
    data['Returns'] = data['Close'].pct_change()
    data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
    data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0

    # Existing indicators
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # New indicators
    data['slowk'], data['slowd'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
    upperband, middleband, lowerband = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['upperband'] = upperband
    data['middleband'] = middleband
    data['lowerband'] = lowerband
    data['MOM'] = talib.MOM(data['Close'], timeperiod=10)

    return data.dropna()

def prepare_data(data):
    """Prepare training and testing datasets for machine learning."""
    data = feature_engineering(data)
    feature_columns = ['RSI', 'ADX', 'OBV', 'MACD', 'MACD_signal', 'ATR', 'slowk', 'slowd', 'CCI', 'upperband', 'middleband', 'lowerband', 'MOM']
    X = data[feature_columns]
    y = (data['Returns'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(stock_code, training_years, model_type='RF'):
    """Train a machine learning model using specified algorithm."""
    data = fetch_data(stock_code, training_years)
    X_train, X_test, y_train, y_test = prepare_data(data)

    if model_type == 'RF':
        model = RandomForestClassifier(random_state=42)
        parameters = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'SVM':
        model = SVC(probability=True)
        parameters = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
            'kernel': ['rbf', 'linear']
        }
    else:
        raise ValueError("Unsupported model type")

    clf = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=-1)
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
