from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import StockPredictionSerializer

class InputView(APIView):
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
