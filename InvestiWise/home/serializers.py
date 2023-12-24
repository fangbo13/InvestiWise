from rest_framework import serializers
from .models import HomeContent

class HomeContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = HomeContent
        fields = ['stock_codes', 'heading', 'home_background']
