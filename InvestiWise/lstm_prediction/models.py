from django.db import models

class LSTMPrediction(models.Model):
    stock_code = models.CharField(max_length=10)  # 股票代码
    training_year = models.IntegerField()  # 训练数据年份
    prediction_days = models.IntegerField()  # 预测天数

    def __str__(self):
        return self.stock_code  # 返回股票代码而不是股票名称

