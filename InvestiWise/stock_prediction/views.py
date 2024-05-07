import yfinance as yf
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .dataprice import fetch_stock_data, summary_statistics
from .ml_module import train_model
from .models import StockData, StockPrediction
from .serializers import StockDataSerializer, StockPredictionSerializer


class InputView(APIView):
    def get(self, request):
            # 查询所有的 StockPrediction 实例
            stock_predictions = StockPrediction.objects.all()
            # 使用序列化器
            serializer = StockPredictionSerializer(stock_predictions, many=True)
            # 返回响应
            return Response(serializer.data)

    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)
        if serializer.is_valid():
            saved_instance = serializer.save()  # 保存实例
            # 从保存的实例调用机器学习模型
            prediction_results = train_model(
                stock_code=saved_instance.stock_code,
                training_years=saved_instance.training_year,
                model_type=saved_instance.ml_model
            )
            # 检查是否有错误返回
            if "error" in prediction_results:
                return Response({"error": prediction_results["error"]}, status=status.HTTP_400_BAD_REQUEST)

            # 成功情况下返回机器学习的结果
            return Response({
                "message": "Data received and prediction made successfully",
                "saved_data": serializer.data,
                "prediction_results": prediction_results
            }, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# Create your views here.
class StockDataView(APIView):
    def get(self, request):
        data = StockData.objects.all()
        serializer = StockDataSerializer(data, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = StockDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            stock_code = request.data.get('stock_code')
            start_date = request.data.get('start_date')
            end_date = request.data.get('end_date')
            stock_data = fetch_stock_data(stock_code, start_date, end_date)
            
            if not stock_data:
                return Response({"error": "No data found"}, status=status.HTTP_404_NOT_FOUND)
            
            stats = summary_statistics(stock_data['prices'])  # Assume stock_data['prices'] is a list of prices

            return Response({
                "message": "Data submitted successfully.",
                "stock_data": stock_data,
                "statistics": stats
            }, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        

class HotStocksDataView(APIView):
    def get(self, request):
        # 热门股票列表
        stock_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "BABA", "NVDA", "PYPL", "NFLX", "BAC","ADBE", "INTC", "CMCSA", "PEP", "CSCO", "AVGO", "TMUS", "QCOM", "TXN", "ABBV","JPM", "V", "HD"]
        data = []
        for symbol in stock_list:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")
            if not hist.empty:
                last_close = hist['Close'].iloc[-1]
                open_price = hist['Open'].iloc[-1]
                change_percent = ((last_close - open_price) / open_price) * 100
                data.append({
                    'symbol': symbol,
                    'last_close': last_close,
                    'change_percent': change_percent
                })
        return Response(data)
    

class MachineLearningView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = StockPredictionSerializer(data=request.data)
        if serializer.is_valid():
            # 从验证后的数据中获取股票信息和其他参数
            stock_code = serializer.validated_data['stock_code']
            training_years = serializer.validated_data['training_year']
            validation_percent = serializer.validated_data['validation_years']
            ml_model = serializer.validated_data['ml_model']
            
            # 调用机器学习模块进行预测
            prediction_results = prepare_and_train(stock_code, training_years, validation_percent, ml_model)
            
            # 返回预测结果
            return Response({
                "message": "Prediction completed successfully.",
                "results": prediction_results
            }, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)    