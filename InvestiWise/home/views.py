from rest_framework.views import APIView
from rest_framework.response import Response
from .models import HomeContent
from .serializers import HomeContentSerializer

class HomeContentView(APIView):
    def get(self, request):
        settings = HomeContent.objects.first()  # 获取第一个实例
        serializer = HomeContentSerializer(settings)
        return Response(serializer.data)
