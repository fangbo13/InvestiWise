import yfinance as yf

def fetch_stock_data(stock_code, start_date, end_date):
    """
    Fetches the closing prices for a given stock within a specified date range.
    """
    data = yf.download(stock_code, start=start_date, end=end_date)
    if data.empty:
        return {}
    closing_prices = data['Close'].tolist()  # 转换为列表以便序列化
    dates = data.index.strftime('%Y-%m-%d').tolist()  # 日期格式化为字符串
    return {
        'dates': dates,
        'prices': closing_prices
    }

def summary_statistics(prices):
    return {
        "max": max(prices),
        "min": min(prices),
        "average": sum(prices) / len(prices)
    }
