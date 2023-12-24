from django.urls import path
from .views import HomeContentView,StockDataView

urlpatterns = [
    path('api/home/', HomeContentView.as_view(),name='home'),
    path('api/stocks/', StockDataView.as_view(),name='stock-data'),

]