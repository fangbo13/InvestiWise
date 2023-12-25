from django.db import models

# Create your models here.

class HomeContent(models.Model):
    stock_codes = models.CharField(max_length=10)  # 用于存储股票代码的文本字段
    heading = models.CharField(max_length=200)
    brand_name = models.CharField(max_length=30, default='', blank=False, null=False)
    introduce = models.CharField(max_length=100)

    def __str__(self):
        return "Home Page Settings"

