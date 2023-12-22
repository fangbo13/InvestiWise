from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from .models import HomeContent
from .serializers import HomeContentSerializer

class HomeContentAPIView(generics.RetrieveAPIView):
    queryset = HomeContent.objects.all()
    serializer_class = HomeContentSerializer
