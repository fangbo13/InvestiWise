from rest_framework.views import APIView
from rest_framework.response import Response
from .models import HomeContent
from .serializers import HomeContentSerializer
import requests
import yfinance as yf

class HomeContentView(APIView):
    def get(self, request):
        settings = HomeContent.objects.first()  # 获取第一个实例
        serializer = HomeContentSerializer(settings)
        return Response(serializer.data)



class StockDataView(APIView):
    def get(self, request):
        stock_list = [
            "AAPL", "MSFT", "BABA", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM",
            "V", "MA", "PYPL", "NFLX", "GOOG", "ADBE", "BAC", "GS", "DIS", "VZ",
            "T", "CRM", "NKE", "CMCSA", "TSM", "MRK", "PFE", "KO", "INTC", "CSCO",
            "ORCL", "WMT", "JNJ", "PG", "UNH", "HD", "TMO", "XOM", "PEP", "NIO",
            "C", "BA", "AAL", "CRM", "CVS", "TSM", "MCD", "TGT", "IBM", "F",
            "GE", "UBER", "LYFT", "GM", "WFC", "ABBV", "ABT", "BMY", "COP", "COST",
            "SBUX", "PBR", "SQ", "PYPL", "TMUS", "NFLX", "CMCSA", "TDOC", "MELI", "NKE",
            "ADBE", "BAC", "GS", "DIS", "VZ", "T", "KO", "INTC", "CSCO", "NOK"
        ]

        data = self.get_stock_data(stock_list)
        return Response(data)

    def get_stock_data(self, stock_list):
        stock_data = {}
        for ticker in stock_list:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                last_close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change_percent = ((last_close - prev_close) / prev_close) * 100
                stock_data[ticker] = f"{ticker} {round(change_percent, 2)}%"
            else:
                stock_data[ticker] = f"{ticker} N/A"
        return stock_data
#  