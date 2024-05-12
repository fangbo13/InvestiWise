from rest_framework import serializers
from .models import StockPrediction

class StockPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockPrediction
        fields = ['stock_code', 'training_year', 'validation_years', 'prediction_days', 'ml_model']
    



from .models import StockData
class StockDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockData
        fields = '__all__'


