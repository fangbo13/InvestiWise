from django.urls import path
from .views import HomeContentView

urlpatterns = [
    path('api/home/', HomeContentView.as_view(),name='home'),
]