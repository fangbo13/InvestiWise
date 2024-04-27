from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import matplotlib.pyplot as plt


class StockPrediction(models.Model):
    # 用户输入的股票代码
    stock_code = models.CharField(max_length=10)

    # 训练数据和验证数据的年份选择
    training_year = models.PositiveIntegerField(
            validators=[
                MinValueValidator(4, message="训练年份必须大于等于4"),
                MaxValueValidator(19, message="训练年份必须小于等于19"),
            ]
        )    
    validation_years = models.PositiveIntegerField()

    # 预测未来走势的天数
    prediction_days = models.PositiveIntegerField()

    # 机器学习模型选择
    ML_MODEL_CHOICES = [
        ('LR', 'Linear Regression'),
        ('RF', 'Random Forest'),
        ('SVM', 'Support Vector Machine'),
        # 可以根据需求添加更多模型
    ]
    ml_model = models.CharField(
        max_length=3,
        choices=ML_MODEL_CHOICES,
        default='LR',
    )

    def __str__(self):
        return f"{self.stock_code} - {self.ml_model}"



class StockData(models.Model):
    stock_code = models.CharField(max_length=10)
    start_date = models.DateField()
    end_date = models.DateField()

    def __str__(self):
        return f"{self.stock_code} from {self.start_date} to {self.end_date}"