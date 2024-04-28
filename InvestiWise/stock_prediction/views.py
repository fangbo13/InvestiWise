import yfinance as yf
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
from .models import StockData, StockPrediction
from .serializers import StockDataSerializer, StockPredictionSerializer
from .dataprice import fetch_stock_data, summary_statistics


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
            # 可选：保存到数据库
            serializer.save()
            # 准备后续处理（例如：提取股票代码）
            # stock_code = serializer.validated_data['stock_code']
            # 返回成功响应
            return Response({"message": "Data received successfully"}, status=status.HTTP_201_CREATED)
        else:
            # 返回错误信息
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
        stock_list = stock_list = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "BABA", "NVDA", "PYPL", "NFLX", "ADBE", "INTC", "CMCSA", "PEP", "CSCO", "AVGO", "TMUS", "QCOM", "TXN", "ABBV"]
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