from django.db import models

# Create your models here.

class HomeContent(models.Model):
    stock_codes = models.TextField()  # 用于存储股票代码的文本字段
    heading = models.CharField(max_length=200)
    home_background = models.ImageField(upload_to='home_background/')
    
    def __str__(self):
        return "Home Content"
