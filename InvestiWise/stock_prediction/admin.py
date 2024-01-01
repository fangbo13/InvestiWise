from django.contrib import admin
from .models import StockPrediction

@admin.register(StockPrediction)
class StockPredictionAdmin(admin.ModelAdmin):
    list_display = ('stock_code', 'stock_name', 'training_year', 'validation_years', 'prediction_days', 'ml_model')
    list_filter = ('ml_model', 'training_year')
    search_fields = ('stock_code', 'stock_name')


