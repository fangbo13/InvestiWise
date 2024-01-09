from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import StockPredictionSerializer
from .models import StockPrediction


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
