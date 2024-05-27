import logging
import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import praw
import talib
import yfinance as yf
from django.http import HttpResponse
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from transformers import pipeline

logger = logging.getLogger(__name__)
# 使用新的API host和API key
openai.api_base = "https://api.chatanywhere.tech"
openai.api_key = "sk-PtqtSxClmAXS2pnJ3doleQHkKQWn6m7XGqiaPqHP0GVUgBdZ"
# Use the Agg backend
plt.switch_backend('Agg')

# Initialize BERT sentiment analyzer
sentiment_analyzer = pipeline('sentiment-analysis')

def fetch_stock_data(stock_code):
    stock = yf.Ticker(stock_code)
    hist = stock.history(period="1y")
    return hist

def fetch_data(stock_code, training_years=10):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=training_years)
    return yf.download(stock_code, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

def calculate_annual_return(data):
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    annual_return = ((end_price - start_price) / start_price) * 100
    return annual_return

def feature_engineering(data, prediction_days=7):
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

def prepare_data(data, prediction_days=7, test_size=0.2):
    data = feature_engineering(data, prediction_days)
    feature_columns = [col for col in data.columns if col not in ['Close', 'Future_Close', 'Target']]
    X = data[feature_columns]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(stock_code):
    data = fetch_data(stock_code, training_years=10)
    X_train, X_test, y_train, y_test = prepare_data(data, prediction_days=7)

    model = SVC(probability=True)
    parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 'scale']}

    clf = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    predictions = best_model.predict(X_test)
    prediction_proba = best_model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_proba)
    roc_auc = roc_auc_score(y_test, prediction_proba)

    return {
        "predictions": predictions,
        "prediction_proba": prediction_proba.tolist(),
        "best_params": clf.best_params_,
        "classification_report": report,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "roc_auc": roc_auc
    }

def create_stock_chart(data, stock_code):
    data['MA10'] = data['Close'].rolling(window=10).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label='Adjusted Close Price')
    plt.plot(data.index, data['MA10'], label='10-Day Moving Average', linestyle='--')
    plt.title(f"{stock_code} Stock Price - Last 1 Year")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # Ensure the image is closed
    return temp_file.name

def create_daily_return_chart(data, stock_code):
    last_month_data = data['Close'].iloc[-30:]
    daily_returns = last_month_data.pct_change().dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_returns.index, daily_returns, label='Daily Return')
    plt.title(f"{stock_code} Daily Return - Last 1 Month")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # Ensure the image is closed
    return temp_file.name, daily_returns

def generate_ma_insights(data):
    current_price = data['Close'].iloc[-1]
    current_ma10 = data['MA10'].iloc[-1]

    if current_price > current_ma10:
        trend = "uptrend"
        position = "above"
    else:
        trend = "downtrend"
        position = "below"

    return current_price, current_ma10, trend, position

def analyze_sentiment(text):
    max_length = 512
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    segment_count = 0
    
    for i in range(0, len(text), max_length):
        segment = text[i:i+max_length]
        sentiment = sentiment_analyzer(segment)
        label = sentiment[0]['label']
        if label == 'POSITIVE':
            sentiment_counts['POSITIVE'] += 1
        elif label == 'NEGATIVE':
            sentiment_counts['NEGATIVE'] += 1
        else:
            sentiment_counts['NEUTRAL'] += 1
        segment_count += 1
    
    # Calculate the predominant sentiment for the entire text
    predominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return predominant_sentiment

    max_length = 512
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    segment_count = 0
    
    for i in range(0, len(text), max_length):
        segment = text[i:i+max_length]
        sentiment = sentiment_analyzer(segment)
        label = sentiment[0]['label']
        if label == 'POSITIVE':
            sentiment_counts['POSITIVE'] += 1
        elif label == 'NEGATIVE':
            sentiment_counts['NEGATIVE'] += 1
        else:
            sentiment_counts['NEUTRAL'] += 1
        segment_count += 1
    
    # Calculate the predominant sentiment for the entire text
    predominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return predominant_sentiment

    max_length = 512
    sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    for i in range(0, len(text), max_length):
        segment = text[i:i+max_length]
        sentiment = sentiment_analyzer(segment)
        label = sentiment[0]['label']
        if label == 'POSITIVE':
            sentiments['POSITIVE'] += 1
        elif label == 'NEGATIVE':
            sentiments['NEGATIVE'] += 1
        else:
            sentiments['NEUTRAL'] += 1
    
    # Print debug information
    print(f"Segment sentiments: {sentiments}")
    
    return sentiments

def get_reddit_sentiments(query):
    reddit = praw.Reddit(
        client_id='ByGHuaBLiK2AdpNTPWKlCA',  
        client_secret='KfB9LAgGXaJ7PhUzRFvNZr32P3g5lg',  
        user_agent='Haibo Fang'  
    )

    total_sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    processed_posts = 0

    for submission in reddit.subreddit('all').search(query, limit=100):
        title = submission.title
        selftext = submission.selftext
        
        text_to_analyze = f"{title}. {selftext}"
        predominant_sentiment = analyze_sentiment(text_to_analyze)
        
        total_sentiments[predominant_sentiment] += 1
        processed_posts += 1

    if processed_posts == 0:
        return 0, 0, 0  # Avoid division by zero
    
    total_positive = (total_sentiments['POSITIVE'] / processed_posts) * 100
    total_negative = (total_sentiments['NEGATIVE'] / processed_posts) * 100
    total_neutral = (total_sentiments['NEUTRAL'] / processed_posts) * 100

    return total_positive, total_negative, total_neutral

    reddit = praw.Reddit(
        client_id='ByGHuaBLiK2AdpNTPWKlCA',  
        client_secret='KfB9LAgGXaJ7PhUzRFvNZr32P3g5lg',  
        user_agent='Haibo Fang'  
    )

    total_sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    processed_posts = 0

    for submission in reddit.subreddit('all').search(query, limit=100):
        title = submission.title
        selftext = submission.selftext
        
        text_to_analyze = f"{title}. {selftext}"
        predominant_sentiment = analyze_sentiment(text_to_analyze)
        
        total_sentiments[predominant_sentiment] += 1
        processed_posts += 1

    if processed_posts == 0:
        return 0, 0, 0  # Avoid division by zero
    
    total_positive = (total_sentiments['POSITIVE'] / processed_posts) * 100
    total_negative = (total_sentiments['NEGATIVE'] / processed_posts) * 100
    total_neutral = (total_sentiments['NEUTRAL'] / processed_posts) * 100

    return total_positive, total_negative, total_neutral

    reddit = praw.Reddit(
        client_id='ByGHuaBLiK2AdpNTPWKlCA',  
        client_secret='KfB9LAgGXaJ7PhUzRFvNZr32P3g5lg',  
        user_agent='Haibo Fang'  
    )

    total_sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    processed_posts = 0

    for submission in reddit.subreddit('all').search(query, limit=100):
        title = submission.title
        selftext = submission.selftext
        
        text_to_analyze = f"{title}. {selftext}"
        sentiments = analyze_sentiment(text_to_analyze)
        
        total_sentiments['POSITIVE'] += sentiments['POSITIVE']
        total_sentiments['NEGATIVE'] += sentiments['NEGATIVE']
        total_sentiments['NEUTRAL'] += sentiments['NEUTRAL']
        
        processed_posts += 1

    if processed_posts == 0:
        return 0, 0  # Avoid division by zero
    
    total_positive = (total_sentiments['POSITIVE'] / processed_posts) * 100
    total_negative = (total_sentiments['NEGATIVE'] / processed_posts) * 100

    # Print debug information
    print(f"Total sentiments: {total_sentiments}")
    print(f"Processed posts: {processed_posts}")
    print(f"Total positive: {total_positive}%, Total negative: {total_negative}%")

    return total_positive, total_negative

    reddit = praw.Reddit(
        client_id='ByGHuaBLiK2AdpNTPWKlCA',  
        client_secret='KfB9LAgGXaJ7PhUzRFvNZr32P3g5lg',  
        user_agent='Haibo Fang'  
    )

    total_sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    processed_posts = 0

    for submission in reddit.subreddit('all').search(query, limit=100):
        title = submission.title
        selftext = submission.selftext
        
        # Combine title and selftext for sentiment analysis
        text_to_analyze = f"{title}. {selftext}"
        sentiments = analyze_sentiment(text_to_analyze)
        
        # Accumulate the sentiment results of each post
        total_sentiments['POSITIVE'] += sentiments['POSITIVE']
        total_sentiments['NEGATIVE'] += sentiments['NEGATIVE']
        total_sentiments['NEUTRAL'] += sentiments['NEUTRAL']
        
        processed_posts += 1

    total_positive = (total_sentiments['POSITIVE'] / processed_posts) * 100
    total_negative = (total_sentiments['NEGATIVE'] / processed_posts) * 100

    return total_positive, total_negative

def create_sentiment_chart(positive, negative, neutral, stock_code):
    
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [positive, negative, neutral]
    colors = ['green', 'red', 'grey']
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f"Market Sentiment for {stock_code}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # Ensure the image is closed
    return temp_file.name

    print(f"Creating sentiment chart with: Positive={positive}%, Negative={negative}%, Neutral={neutral}%")
    
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [positive, negative, neutral]
    colors = ['green', 'red', 'grey']
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f"Market Sentiment for {stock_code}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # Ensure the image is closed
    return temp_file.name

    print(f"Creating sentiment chart with: Positive={positive}%, Negative={negative}%")
    
    labels = 'Positive', 'Negative'
    sizes = [positive, negative]
    colors = ['green', 'red']
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f"Market Sentiment for {stock_code}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # Ensure the image is closed
    return temp_file.name

def create_title(stock_code):
    drawing = Drawing(500, 50)
    drawing.add(String(250, 40, f"Investment Report for {stock_code}", fontSize=24, fillColor=colors.black, textAnchor='middle'))
    drawing.add(Line(50, 30, 450, 30, strokeColor=colors.black, strokeWidth=1))
    drawing.add(String(450, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", fontSize=12, fillColor=colors.grey, textAnchor='end'))
    return drawing

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

def create_lstm_chart(data, predictions, y_test, test_predictions, prediction_dates, future_predictions):

    plt.figure(figsize=(16, 6))
    plt.title('LSTM Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(data['dates'], data['historical'], label='Historical')
    plt.plot(data['dates'][-len(y_test):], y_test, label='Actual')
    plt.plot(data['dates'][-len(test_predictions):], test_predictions, label='Predictions')
    plt.plot(prediction_dates, future_predictions, label='Future Predictions')
    plt.legend(['Historical', 'Actual', 'Predictions', 'Future Predictions'], loc='lower right')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()
    return temp_file.name

def create_zoomed_lstm_chart(data, prediction_dates, future_predictions):

    zoom_days = 20
    total_days = zoom_days + len(future_predictions)
    
    # 确保索引范围有效
    recent_dates = data['dates'][-(total_days + 1):]  # 确保包含足够的日期
    recent_actual = data['historical'][-(total_days + 1):]
    recent_test_predictions = data['test_predictions'][-zoom_days:]

    plt.figure(figsize=(16, 6))
    plt.title('Zoomed LSTM Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(recent_dates[:zoom_days], recent_actual[:zoom_days], label='Actual (Historical)')
    plt.plot(recent_dates[:zoom_days], recent_test_predictions, label='Predictions (Validation)')
    plt.plot(prediction_dates, future_predictions, label='Future Predictions')
    all_dates = recent_dates[:zoom_days] + prediction_dates
    plt.xticks(rotation=45, ha='right')
    plt.xticks(ticks=np.arange(0, len(all_dates), step=5), labels=[all_dates[i] for i in range(0, len(all_dates), 5)])
    plt.legend(['Actual (Historical)', 'Predictions (Validation)', 'Future Predictions'], loc='lower right')
    plt.grid(True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()
    return temp_file.name

def generate_gpt_content(stock_code, last_close_price, annual_return, current_price, current_ma10, trend, position, volatility, avg_daily_return, last_5_days_returns, future_predictions, positive_sentiment, negative_sentiment):
    prompt = f"""
    You are a professional financial analyst. Based on the following data, generate a concise stock investment recommendation for {stock_code}. Ensure the recommendation is no more than 250 words.

    Closing Price: ${last_close_price:.2f}
    Annual Return: {annual_return:.2f}%
    Current Price: ${current_price:.2f}
    10-Day Moving Average: ${current_ma10:.2f}
    Trend: {trend}
    Position: {position}
    Volatility (Last 30 Days): {volatility:.2f}%
    Average Daily Return (Last 30 Days): {avg_daily_return:.2f}%
    Last 5 Days Returns: {last_5_days_returns}
    Positive Sentiment: {positive_sentiment:.2f}%
    Negative Sentiment: {negative_sentiment:.2f}%
    LSTM Prediction (Next 7 Days): {future_predictions}

    Based on the analysis, here is the investment recommendation for {stock_code}:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional financial analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message['content'].strip()
    
def generate_pdf_report(stock_code):
    try:
        stock_data = fetch_stock_data(stock_code)
        chart_path = create_stock_chart(stock_data, stock_code)
        daily_return_chart_path, daily_returns = create_daily_return_chart(stock_data, stock_code)
        last_close_price = stock_data['Close'].iloc[-1]
        annual_return = calculate_annual_return(stock_data)
        current_price, current_ma10, trend, position = generate_ma_insights(stock_data)

        total_positive, total_negative, total_neutral = get_reddit_sentiments(stock_code)
        sentiment_chart_path = create_sentiment_chart(total_positive, total_negative, total_neutral, stock_code)

        # Call the LSTM model for prediction results
        lstm_results = predict_future_prices(stock_code)
        lstm_chart_path = create_lstm_chart(lstm_results, lstm_results['predictions'], lstm_results['test'], lstm_results['test_predictions'], lstm_results['prediction_dates'], lstm_results['predictions'])
        zoomed_lstm_chart_path = create_zoomed_lstm_chart(lstm_results, lstm_results['prediction_dates'], lstm_results['predictions'])

        # Calculate volatility and average daily return
        volatility = daily_returns.std() * 100
        avg_daily_return = daily_returns.mean() * 100

        # Collect last 5 days returns
        last_5_days_returns = {str(date.date()): ret * 100 for date, ret in daily_returns[-5:].items()}

        # Generate content using GPT
        future_predictions = {date: price for date, price in zip(lstm_results['prediction_dates'], lstm_results['predictions'])}
        gpt_content = generate_gpt_content(stock_code, last_close_price, annual_return, current_price, current_ma10, trend, position, volatility, avg_daily_return, last_5_days_returns, future_predictions, total_positive, total_negative)

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{stock_code}_report.pdf"'

        doc = SimpleDocTemplate(response, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        subtitle_style = styles['Heading2']
        normal_style = styles['BodyText']
        gpt_style = ParagraphStyle('GPT', parent=normal_style, fontSize=10, spaceBefore=10, spaceAfter=10)

        # Title
        title = create_title(stock_code)
        elements.append(title)
        elements.append(Spacer(1, 5))  # Reduce space between title and top of the page

        # Image
        elements.append(Paragraph(f"Stock Price Chart for {stock_code}", subtitle_style))
        elements.append(Spacer(1, 5))  # Reduce space between title and image
        img = Image(chart_path)
        img.drawHeight = 3 * inch
        img.drawWidth = 6 * inch
        elements.append(img)
        elements.append(Spacer(1, 20))  # Add a blank line

        # Table
        data = [
            ["Stock Code", stock_code],
            ["Closing Price on", f"${last_close_price:.2f}"],
            ["Annual Return", f"{annual_return:.2f}%"],
            ["Current Price", f"${current_price:.2f}"],
            ["10-Day Moving Average", f"${current_ma10:.2f}"],
            ["Trend", f"The stock is currently in a {trend} and is {position} the 10-day moving average."]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 5))  # Add a 5px blank line

        # MA10 explanation
        explanation = """
        <b>Explanation:</b><br/>
        The 10-day moving average (MA10) is the average closing price over the past 10 days.
        It helps to smooth out price data and identify trends.
        If the current price is above the MA10, it indicates an uptrend.
        If it's below, it indicates a downtrend.
        """
        elements.append(Paragraph(explanation, normal_style))

        # Add a page break
        elements.append(PageBreak())

        # Market analysis chart and information
        elements.append(Paragraph(f"Market Analysis for {stock_code}", subtitle_style))
        elements.append(Spacer(1, 2))  # Reduce space between title and top of the page

        # Daily Return chart
        daily_return_img = Image(daily_return_chart_path)
        daily_return_img.drawHeight = 3 * inch
        daily_return_img.drawWidth = 6 * inch
        elements.append(daily_return_img)
        elements.append(Spacer(1, 20))  # Add a blank line

        # Daily Return explanation
        daily_return_explanation = """
        <b>Daily Return Explanation:</b><br/>
        The daily return is calculated as the percentage change in the stock's closing price from one day to the next. 
        It helps investors understand the stock's short-term performance and volatility.<br/><br/>
        The formula for daily return is:<br/>
        <b><i>Daily Return = (Closing Price on Current Day - Closing Price on Previous Day) / Closing Price on Previous Day</i></b><br/><br/>
        Below are some example daily returns for the last 5 days:<br/>
        """
        daily_return_paragraph = Paragraph(daily_return_explanation, normal_style)

        # Add example data
        example_data = [[str(date.date()), f"{ret*100:.2f}%"] for date, ret in daily_returns[-5:].items()]
        example_table = Table([["Date", "Daily Return"]] + example_data)
        example_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        # Add more useful information
        additional_info = """
        <b>Volatility:</b> The standard deviation of daily returns over the past month is an indicator of the stock's volatility.<br/>
        """
        additional_info_paragraph = Paragraph(additional_info, normal_style)
        
        # Calculate and add more data
        volatility = daily_returns.std() * 100
        average_daily_return = daily_returns.mean() * 100

        additional_data = [
            ["Volatility (Last 30 Days)", f"{volatility:.2f}%"],
            ["Average Daily Return (Last 30 Days)", f"{average_daily_return:.2f}%"]
        ]
        additional_table = Table(additional_data)
        additional_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        # Create a side-by-side layout for the table and pie chart
        sentiment_chart_img = Image(sentiment_chart_path)
        sentiment_chart_img.drawHeight = 2.5 * inch
        sentiment_chart_img.drawWidth = 2.5 * inch
        sentiment_explanation = f"""
        <b>Sentiment Analysis Explanation:</b><br/>
        Based on recent Reddit posts mentioning {stock_code}, the market sentiment is as follows:<br/>
        Positive: {total_positive:.2f}%, Negative: {total_negative:.2f}%
        """
        sentiment_paragraph = Paragraph(sentiment_explanation, normal_style)

        right_column_content = [
            [additional_info_paragraph],
            [Spacer(1, 5)],  # Add a 5px blank line
            [additional_table],
            [sentiment_chart_img],
            [sentiment_paragraph]
        ]

        two_column_table = Table(
            [
                [
                    [daily_return_paragraph, Spacer(1, 12), example_table],  # Remove daily_return_table_name
                    Spacer(1, 0.5 * inch),  # Increase space between the tables
                    right_column_content  # Include right column content
                ]
            ],
            colWidths=[doc.width / 2.0 - 30, 30, doc.width / 2.0 - 30]  # Adjust column widths
        )
        two_column_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.transparent),  # Set the container border to transparent
            ('BOX', (0, 0), (-1, -1), 0.25, colors.transparent),  # Set the container border to transparent
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10)
        ]))

        elements.append(two_column_table)
        elements.append(Spacer(1, 20))  # Add a blank line

        # LSTM prediction results
        elements.append(PageBreak())
        elements.append(Paragraph(f"LSTM Prediction Analysis for {stock_code}", subtitle_style))
        elements.append(Spacer(1, 5))  # Reduce space between title and image
        lstm_img = Image(lstm_chart_path)
        lstm_img.drawHeight = 3 * inch
        lstm_img.drawWidth = 6 * inch
        elements.append(lstm_img)

        lstm_explanation = """
        <b>LSTM Prediction Results:</b><br/>
        The chart below shows the LSTM model's predictions for the stock prices over the last 20 days and the next 7 days.
        """
        elements.append(Paragraph(lstm_explanation, normal_style))

        # Adding LSTM explanation
        lstm_details = """
        <b>Understanding LSTM Predictions:</b><br/>
        LSTM (Long Short-Term Memory) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.
        The chart above shows the LSTM model's ability to predict stock prices based on historical data.<br/><br/>
        <b>Mean Squared Error (MSE):</b> The MSE is a measure of the average squared difference between the predicted and actual values. 
        It indicates how close the predicted prices are to the actual prices.<br/>
        """
        elements.append(Paragraph(lstm_details, normal_style))

        # Calculate and display MSE
        mse = np.mean((np.array(lstm_results['test_predictions']) - np.array(lstm_results['test']))**2)
        elements.append(Spacer(1, 7))  # Add 7px space before MSE
        elements.append(Paragraph(f"<b>MSE:</b> {mse:.4f}", normal_style))
        
        elements.append(Spacer(1, 10))  # Add 10px space before the table

        # Add last 7 days data table
        last_7_days_data = [[lstm_results['dates'][-7:][i], lstm_results['test'][-7:][i], lstm_results['test_predictions'][-7:][i]] for i in range(7)]
        last_7_days_table = Table([["Date", "Actual", "Predicted"]] + last_7_days_data)
        last_7_days_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(last_7_days_table)

        # Zoomed LSTM prediction results on the next page with a small title
        elements.append(PageBreak())
        elements.append(Paragraph(f"Detailed LSTM Predictions", subtitle_style))
        elements.append(Spacer(1, 5))  # Reduce space between title and image
        zoomed_lstm_img = Image(zoomed_lstm_chart_path)
        zoomed_lstm_img.drawHeight = 3 * inch
        zoomed_lstm_img.drawWidth = 6 * inch
        elements.append(zoomed_lstm_img)

        # Small title before GPT content
        elements.append(Spacer(1, 20))  # Add a blank line
        elements.append(Paragraph("<b>Insights and Future Predictions:</b>", ParagraphStyle('Bold', parent=normal_style, fontSize=12, spaceBefore=10, spaceAfter=10)))
        elements.append(Spacer(1, 5))  # Reduce space between title and content
        
        gpt_paragraphs = gpt_content.split("\n\n")
        for paragraph in gpt_paragraphs:
            elements.append(Paragraph(paragraph, gpt_style))
            elements.append(Spacer(1, 5))

        doc.build(elements)

        # Remove temporary files
        os.remove(chart_path)
        os.remove(daily_return_chart_path)
        os.remove(sentiment_chart_path)
        os.remove(lstm_chart_path)
        os.remove(zoomed_lstm_chart_path)
    
        return response
    except Exception as e:
        logger.error(f"Error generating PDF report for stock code {stock_code}: {e}")
        raise
