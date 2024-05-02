from rest_framework import serializers
from .models import StockPrediction

class StockPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockPrediction
        fields = ['stock_code', 'training_year', 'validation_years', 'prediction_days', 'ml_model']
    
    def validate_training_year(self, value):
        if value < 4 or value > 21:
            raise serializers.ValidationError("训练年份必须在4至19之间。")
        return value




from .models import StockData
class StockDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockData
        fields = '__all__'


