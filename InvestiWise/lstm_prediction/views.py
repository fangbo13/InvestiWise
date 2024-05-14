from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
import yfinance as yf
import pandas as pd
from .models import LSTMPrediction
from .serializers import LSTMSerializer
from .ml_module import train_model  # 假设你已经有了 lstm_predict 函数

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
