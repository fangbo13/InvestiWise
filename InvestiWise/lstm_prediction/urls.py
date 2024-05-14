from django.urls import path
from .views import LSTMInputView
from .views import moving_average_view

urlpatterns = [
    path('predict_lstm/', LSTMInputView.as_view(), name='predict_lstm'),
    path('moving-average/', moving_average_view, name='moving_average')
]
