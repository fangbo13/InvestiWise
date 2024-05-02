from django.urls import path

from .views import HotStocksDataView, InputView, StockDataView

urlpatterns = [
    path('StockForm/', InputView.as_view(),name='StockForm'),
    path('stock_data/', StockDataView.as_view(), name='stock_data'),
    path('hot-stocks/', HotStocksDataView.as_view(), name='hot_stocks'),
]