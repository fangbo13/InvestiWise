from rest_framework import serializers

from .models import LSTMPrediction


class LSTMSerializer(serializers.ModelSerializer):
    class Meta:
        model = LSTMPrediction
        fields = '__all__'