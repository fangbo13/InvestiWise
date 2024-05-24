import datetime
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf
from django.http import HttpResponse, JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .lstm_model import get_trained_lstm_model, predict_future_prices
from .ml_module import train_model
from .models import LSTMPrediction
from .serializers import LSTMSerializer
from .stock_report import generate_pdf_report


class LSTMInputView(APIView):
    def post(self, request):
        # 使用 LSTM 序列化器处理传入数据
        serializer = LSTMSerializer(data=request.data)
        if serializer.is_valid():
            saved_instance = serializer.save()  # 保存实例
            # 调用 LSTM 预测模型
            prediction_results = train_model(
                stock_code=saved_instance.stock_code,
                training_years=saved_instance.training_year,
                prediction_days=saved_instance.prediction_days
            )
            # 检查预测结果中是否有错误
            if "error" in prediction_results:
                return Response({"error": prediction_results["error"]}, status=status.HTTP_400_BAD_REQUEST)

            # 成功情况下返回 LSTM 预测的结果
            return Response({
                "message": "Data received and LSTM prediction made successfully",
                "saved_data": serializer.data,
                "prediction_results": prediction_results
            }, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def get_stock_prices(stock_code, start_date, end_date):
    data = yf.download(stock_code, start=start_date, end=end_date)
    return data

def moving_average_view(request):
    stock_code = request.GET.get('stock_code', 'AAPL')
    start_date = request.GET.get('start_date', '2020-01-01')
    end_date = request.GET.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
    window = int(request.GET.get('window', 30))
    data = get_stock_prices(stock_code, start_date, end_date)
    data['Moving Average'] = data['Adj Close'].rolling(window=window).mean()
    data_json = data[['Adj Close', 'Moving Average']].dropna().to_json(date_format='iso')
    return JsonResponse({'data': data_json})


def daily_return_view(request):
    stock_code = request.GET.get('stock_code', 'AAPL')
    start_date = request.GET.get('start_date', '2020-01-01')
    end_date = request.GET.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
    data = get_stock_prices(stock_code, start_date, end_date)
    data['Daily Return'] = data['Adj Close'].pct_change()
    data_json = data[['Adj Close', 'Daily Return']].dropna().to_json(date_format='iso')
    return JsonResponse({'data': data_json})


@api_view(['GET'])
def get_risk_coefficients(request):
    stocks = ['AAPL', 'TSLA', 'EBAY', 'MSFT', 'GOOGL', 'AMZN', 'NFLX', 'NVDA', 'AMD', 'INTC', 'PYPL', 'CSCO', 'META', 'CRM', 'IBM', 'QCOM']
# 添加更多的股票代码
    data = {}

    for stock in stocks:
        df = yf.download(stock, period='1y')
        df['Daily Return'] = df['Close'].pct_change()
        returns = df['Daily Return'].dropna()
        risk_coefficient = returns.std()
        data[stock] = risk_coefficient

    return Response(data)


@api_view(['GET'])
def calculate_risk(request):
    start_date = request.GET.get('startDate')
    end_date = request.GET.get('endDate')
    stock_code = request.GET.get('stockCode')

    if not start_date or not end_date or not stock_code:
        return Response({'error': 'Start date, end date and stock code are required'}, status=400)

    try:
        df = yf.download(stock_code, start=start_date, end=end_date)
        if df.empty:
            return Response({'error': 'No data found for the given date range and stock code'}, status=400)
        df['Daily Return'] = df['Close'].pct_change()
        returns = df['Daily Return'].dropna()
        if returns.empty:
            return Response({'error': 'Not enough data to calculate risk'}, status=400)
        risk_coefficient = returns.std()
        return Response({'risk_coefficient': risk_coefficient})
    except Exception as e:
        return Response({'error': str(e)}, status=400)
    

@api_view(['GET'])
def predict_stock_price(request):
    stock_code = request.GET.get('stockCode', 'AAPL')
    days_to_predict = int(request.GET.get('days', 7))  # 默认预测7天
    
    result = predict_future_prices(stock_code=stock_code, days_to_predict=days_to_predict)
    
    return JsonResponse(result)


from .utils import get_reddit_sentiments


@api_view(['GET'])
def get_sentiment(request):
    query = request.query_params.get('query')
    if not query or len(query) > 100:
        return Response({'error': 'Query parameter is required and should be less than 100 characters'}, status=400)

    sentiment_data = get_reddit_sentiments(query)
    return Response(sentiment_data)



logger = logging.getLogger(__name__)

class GenerateStockReportView(APIView):
    def post(self, request):
        stock_code = request.data.get('stock_code')
        if not stock_code:
            return Response({"error": "Stock code is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            return generate_pdf_report(stock_code)
        except Exception as e:  
            logger.error(f"Error in GenerateStockReportView: {e}")
            return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
