from django.db import models

class StockPrediction(models.Model):
    # 用户输入的股票代码
    stock_code = models.CharField(max_length=10)
    stock_name = models.CharField(max_length=100)

    # 训练数据和验证数据的年份选择
    training_year = models.IntegerField()
    validation_years = models.IntegerField()

    # 预测未来走势的天数
    prediction_days = models.IntegerField()

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



