from django.urls import path

from .views import HotStocksDataView, InputView, StockDataView

urlpatterns = [
    path('StockPrediction/', InputView.as_view(),name='StockPrediction'),
    path('stock_data/', StockDataView.as_view(), name='stock_data'),
    path('hot-stocks/', HotStocksDataView.as_view(), name='hot_stocks'),
]