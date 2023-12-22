from django.contrib import admin

# Register your models here.
from .models import HomeContent

@admin.register(HomeContent)
class HomeContentAdmin(admin.ModelAdmin):
    list_display = ['stock_codes', 'home_background', 'heading']
    list_display_links = list_display
