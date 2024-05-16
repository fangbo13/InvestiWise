import numpy as np
import pandas as pd
import yfinance as yf
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .ml_module import train_model  # 假设你已经有了 lstm_predict 函数
from .models import LSTMPrediction
from .serializers import LSTMSerializer

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