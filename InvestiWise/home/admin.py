from django.contrib import admin

# Register your models here.
from .models import HomeContent

@admin.register(HomeContent)
class HomeContentAdmin(admin.ModelAdmin):
    list_display = ['stock_codes', 'brand_name', 'heading','introduce']
    list_display_links = list_display
