from django.urls import path
from .views import HomeContentView,StockDataView

urlpatterns = [
    path('home/', HomeContentView.as_view(),name='home'),
    path('stocks/', StockDataView.as_view(),name='stock-data'),

]