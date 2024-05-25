import logging
import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import praw
import yfinance as yf
from django.http import HttpResponse
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from transformers import pipeline

logger = logging.getLogger(__name__)

# 使用Agg后端
plt.switch_backend('Agg')

# 初始化BERT情感分析器
sentiment_analyzer = pipeline('sentiment-analysis')

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
    data['MA10'] = data['Close'].rolling(window=10).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label='Adjusted Close Price')
    plt.plot(data.index, data['MA10'], label='10-Day Moving Average', linestyle='--')
    plt.title(f"{stock_code} Stock Price - Last 1 Year")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # 确保图像关闭
    return temp_file.name

def create_daily_return_chart(data, stock_code):
    last_month_data = data['Close'].iloc[-30:]
    daily_returns = last_month_data.pct_change().dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_returns.index, daily_returns, label='Daily Return')
    plt.title(f"{stock_code} Daily Return - Last 1 Month")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # 确保图像关闭
    return temp_file.name, daily_returns

def generate_ma_insights(data):
    current_price = data['Close'].iloc[-1]
    current_ma10 = data['MA10'].iloc[-1]

    if current_price > current_ma10:
        trend = "uptrend"
        position = "above"
    else:
        trend = "downtrend"
        position = "below"

    return current_price, current_ma10, trend, position

def analyze_sentiment(text):
    max_length = 512
    sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    
    # 分段处理
    for i in range(0, len(text), max_length):
        segment = text[i:i+max_length]
        sentiment = sentiment_analyzer(segment)
        label = sentiment[0]['label']
        if label == 'POSITIVE':
            sentiments['POSITIVE'] += 1
        elif label == 'NEGATIVE':
            sentiments['NEGATIVE'] += 1
        else:
            sentiments['NEUTRAL'] += 1
    
    return sentiments

def get_reddit_sentiments(query):
    reddit = praw.Reddit(
        client_id='ByGHuaBLiK2AdpNTPWKlCA',  
        client_secret='KfB9LAgGXaJ7PhUzRFvNZr32P3g5lg',  
        user_agent='Haibo Fang'  
    )

    total_sentiments = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    processed_posts = 0

    for submission in reddit.subreddit('all').search(query, limit=100):
        title = submission.title
        selftext = submission.selftext
        
        # 合并标题和自文本进行情感分析
        text_to_analyze = f"{title}. {selftext}"
        sentiments = analyze_sentiment(text_to_analyze)
        
        # 累加每篇帖子的情感结果
        total_sentiments['POSITIVE'] += sentiments['POSITIVE']
        total_sentiments['NEGATIVE'] += sentiments['NEGATIVE']
        total_sentiments['NEUTRAL'] += sentiments['NEUTRAL']
        
        processed_posts += 1

    total_positive = (total_sentiments['POSITIVE'] / processed_posts) * 100
    total_negative = (total_sentiments['NEGATIVE'] / processed_posts) * 100

    return total_positive, total_negative

def create_sentiment_chart(positive, negative, stock_code):
    labels = 'Positive', 'Negative'
    sizes = [positive, negative]
    colors = ['green', 'red']
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title(f"Market Sentiment for {stock_code}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, format='png')
    plt.close()  # 确保图像关闭
    return temp_file.name

def create_title(stock_code):
    drawing = Drawing(500, 50)
    drawing.add(String(250, 40, f"Investment Report for {stock_code}", fontSize=24, fillColor=colors.black, textAnchor='middle'))
    drawing.add(Line(50, 30, 450, 30, strokeColor=colors.black, strokeWidth=1))
    drawing.add(String(450, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", fontSize=12, fillColor=colors.grey, textAnchor='end'))
    return drawing

def generate_pdf_report(stock_code):
    try:
        stock_data = fetch_stock_data(stock_code)
        chart_path = create_stock_chart(stock_data, stock_code)
        daily_return_chart_path, daily_returns = create_daily_return_chart(stock_data, stock_code)
        last_close_price = stock_data['Close'].iloc[-1]
        annual_return = calculate_annual_return(stock_data)
        current_price, current_ma10, trend, position = generate_ma_insights(stock_data)

        total_positive, total_negative = get_reddit_sentiments(stock_code)
        sentiment_chart_path = create_sentiment_chart(total_positive, total_negative, stock_code)

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{stock_code}_report.pdf"'

        doc = SimpleDocTemplate(response, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        subtitle_style = styles['Heading2']
        normal_style = styles['BodyText']

        # 标题
        title = create_title(stock_code)
        elements.append(title)
        elements.append(Spacer(1, 5))  # 缩小标题与页面顶部的间距

        # 图像
        elements.append(Paragraph(f"Stock Price Chart for {stock_code}", subtitle_style))
        elements.append(Spacer(1, 5))  # 缩小标题与图片之间的间距
        img = Image(chart_path)
        img.drawHeight = 3 * inch
        img.drawWidth = 6 * inch
        elements.append(img)
        elements.append(Spacer(1, 20))  # 添加空白行

        # 表格
        data = [
            ["Stock Code", stock_code],
            ["Closing Price on", f"${last_close_price:.2f}"],
            ["Annual Return", f"{annual_return:.2f}%"],
            ["Current Price", f"${current_price:.2f}"],
            ["10-Day Moving Average", f"${current_ma10:.2f}"],
            ["Trend", f"The stock is currently in a {trend} and is {position} the 10-day moving average."]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 5))  # 添加5px的空白行

        # MA10解释
        explanation = """
        <b>Explanation:</b><br/>
        The 10-day moving average (MA10) is the average closing price over the past 10 days.
        It helps to smooth out price data and identify trends.
        If the current price is above the MA10, it indicates an uptrend.
        If it's below, it indicates a downtrend.
        """
        elements.append(Paragraph(explanation, normal_style))

        # 添加分页符
        elements.append(PageBreak())

        # Daily Return 图表
        elements.append(Spacer(1, 2))   # 缩小与页面顶部的间距
        elements.append(Paragraph(f"Daily Return Chart for {stock_code}", subtitle_style))
        elements.append(Spacer(1, 2))   # 缩小标题与图片之间的间距
        daily_return_img = Image(daily_return_chart_path)
        daily_return_img.drawHeight = 3 * inch
        daily_return_img.drawWidth = 6 * inch
        elements.append(daily_return_img)
        elements.append(Spacer(1, 20))  # 添加空白行

        # Daily Return 解释
        daily_return_explanation = """
        <b>Daily Return Explanation:</b><br/>
        The daily return is calculated as the percentage change in the stock's closing price from one day to the next. 
        It helps investors understand the stock's short-term performance and volatility.<br/><br/>
        The formula for daily return is:<br/>
        <b><i>Daily Return = (Closing Price on Current Day - Closing Price on Previous Day) / Closing Price on Previous Day</i></b><br/><br/>
        Below are some example daily returns for the last 5 days:<br/>
        """
        daily_return_paragraph = Paragraph(daily_return_explanation, normal_style)

        # 添加示例数据
        example_data = [[str(date.date()), f"{ret*100:.2f}%"] for date, ret in daily_returns[-5:].items()]
        example_table = Table([["Date", "Daily Return"]] + example_data)
        example_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        # 添加更多有用的信息
        additional_info = """
        <b>Volatility:</b> The standard deviation of daily returns over the past month is an indicator of the stock's volatility.<br/>
        """
        additional_info_paragraph = Paragraph(additional_info, normal_style)
        
        # 计算并添加更多数据
        volatility = daily_returns.std() * 100
        average_daily_return = daily_returns.mean() * 100

        additional_data = [
            ["Volatility (Last 30 Days)", f"{volatility:.2f}%"],
            ["Average Daily Return (Last 30 Days)", f"{average_daily_return:.2f}%"]
        ]
        additional_table = Table(additional_data)
        additional_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        # 创建并列布局的表格和饼图
        sentiment_chart_img = Image(sentiment_chart_path)
        sentiment_chart_img.drawHeight = 2.5 * inch
        sentiment_chart_img.drawWidth = 2.5 * inch
        sentiment_explanation = f"""
        <b>Sentiment Analysis Explanation:</b><br/>
        Based on recent Reddit posts mentioning {stock_code}, the market sentiment is as follows:<br/>
        Positive: {total_positive:.2f}%, Negative: {total_negative:.2f}%
        """
        sentiment_paragraph = Paragraph(sentiment_explanation, normal_style)

        right_column_content = [
            [additional_info_paragraph],
            [Spacer(1, 5)],  # 添加5px的空白行
            [additional_table],
            [sentiment_chart_img],
            [sentiment_paragraph]
        ]

        two_column_table = Table(
            [
                [
                    [daily_return_paragraph, Spacer(1, 12), example_table],  # 删除 daily_return_table_name
                    Spacer(1, 0.5 * inch),  # 增加左右表格之间的间距
                    right_column_content  # 将右列内容包括在内
                ]
            ],
            colWidths=[doc.width / 2.0 - 30, 30, doc.width / 2.0 - 30]  # 调整列宽
        )
        two_column_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.transparent),  # 将容器的边框设为透明
            ('BOX', (0, 0), (-1, -1), 0.25, colors.transparent),  # 将容器的边框设为透明
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10)
        ]))

        elements.append(two_column_table)
        elements.append(Spacer(1, 20))  # 添加空白行

        doc.build(elements)

        # 删除临时文件
        os.remove(chart_path)
        os.remove(daily_return_chart_path)
        os.remove(sentiment_chart_path)
    
        return response
    except Exception as e:
        logger.error(f"Error generating PDF report for stock code {stock_code}: {e}")
        raise
