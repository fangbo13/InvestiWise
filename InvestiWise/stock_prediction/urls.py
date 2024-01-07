from django.urls import path
from .views import InputView

urlpatterns = [
    path('api/StockPrediction/', InputView.as_view(),name='StockPrediction'),
]