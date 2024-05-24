import logging
import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import yfinance as yf
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

# 使用Agg后端
plt.switch_backend('Agg')

def fetch_stock_data(stock_code):
    stock = yf.Ticker(stock_code)
    hist = stock.history(period="1y")
    return hist

def calculate_annual_return(data):
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    annual_return = ((end_price - start_price) / start_price) * 100
    return annual_return

def create_stock_chart(data, stock_code):
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'])
    plt.title(f"{stock_code} Stock Price - Last 1 Year")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # 确保图像关闭
    return temp_file.name

def generate_pdf_report(stock_code):
    try:
        stock_data = fetch_stock_data(stock_code)
        chart_path = create_stock_chart(stock_data, stock_code)
        last_close_price = stock_data['Close'].iloc[-1]
        annual_return = calculate_annual_return(stock_data)

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{stock_code}_report.pdf"'

        c = canvas.Canvas(response, pagesize=letter)
        width, height = letter

        # 标题
        c.setFont("Helvetica-Bold", 18)
        c.drawString(100, height - 50, f"Investment Report for {stock_code}")
        c.line(100, height - 55, width - 100, height - 55)

        # 绘制图像 (调整X轴位置、宽度和高度)
        image_x = 50  # 左移图像
        image_y = height - 400
        image_width = 500
        image_height = 300
        c.drawImage(chart_path, image_x, image_y, width=image_width, height=image_height)

        # 绘制文本内容
        c.setFont("Helvetica", 12)
        text = c.beginText(50, image_y - 30)  # 位置在图像下方
        text.setFont("Helvetica", 12)
        
        # 第一行
        text.textOut("Stock Code: ")
        text.setFont("Helvetica-Bold", 12)
        text.textLine(f"{stock_code}")

        # 第二行
        text.setFont("Helvetica", 12)
        text.textOut(f"Closing Price on {datetime.now().strftime('%Y-%m-%d')}: ")
        text.setFont("Helvetica-Bold", 12)
        text.textLine(f"${last_close_price:.2f}")

        # 第三行
        text.setFont("Helvetica", 12)
        text.textOut("Annual Return: ")
        text.setFont("Helvetica-Bold", 12)
        text.textLine(f"{annual_return:.2f}%")

        c.drawText(text)

        c.save()

        # 删除临时文件
        os.remove(chart_path)

        return response
    except Exception as e:
        logger.error(f"Error generating PDF report for stock code {stock_code}: {e}")
        raise
