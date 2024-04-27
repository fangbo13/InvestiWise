from django.urls import path
from .views import InputView,StockDataView

urlpatterns = [
    path('StockPrediction/', InputView.as_view(),name='StockPrediction'),
    path('stock_data/', StockDataView.as_view(), name='stock_data'),

]