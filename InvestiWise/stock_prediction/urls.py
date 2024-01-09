from django.urls import path
from .views import InputView

urlpatterns = [
    path('StockPrediction/', InputView.as_view(),name='StockPrediction'),
]