# import yfinance as yf
# import requests
# from config.config import config
# from transformers import pipeline
# from statistics import mode

# class RecommendationEngine:
#     def __init__(self):
#         self.news_api_key = config.NEWS_API_KEY
#         self.twitter_bearer_token = config.TWITTER_BEARER_TOKEN
#         self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

#     def fetch_news_headlines(self, symbol):
#         url = (
#             f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}&language=en&pageSize=5"
#         )
#         response = requests.get(url)
#         if response.status_code == 200:
#             return [article["title"] for article in response.json().get("articles", [])]
#         return []

#     def fetch_twitter_posts(self, symbol):
#         query = f"{symbol} stock -is:retweet lang:en"
#         url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10"
#         headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
#         response = requests.get(url, headers=headers)
#         if response.status_code == 200:
#             return [tweet["text"] for tweet in response.json().get("data", [])]
#         return []

#     def analyze_sentiment(self, texts):
#         if not texts:
#             return "Neutral"
#         results = self.sentiment_pipeline(texts)
#         sentiments = [result["label"] for result in results]
#         try:
#             return mode(sentiments)  # Use majority vote
#         except:
#             return "Neutral"

#     def get_price_trend(self, symbol):
#         try:
#             stock = yf.Ticker(symbol)
#             hist = stock.history(period="7d", interval="1d")  # 7-day trend
#             closing_prices = hist["Close"].values
#             if len(closing_prices) < 2:
#                 return "neutral"
#             trend = np.polyfit(range(len(closing_prices)), closing_prices, 1)[0]
#             if trend > 0:
#                 return "increasing"
#             elif trend < 0:
#                 return "decreasing"
#             else:
#                 return "neutral"
#         except Exception as e:
#             print(f"[Trend Error] {e}")
#             return "neutral"

#     def get_recommendation(self, symbol, market_info, sentiment=None, user_profile=None):
#         price = market_info["current_price"]
#         trend = market_info.get("trend", "neutral")  # This should be "increasing", "decreasing", or "neutral"

#         news = self.fetch_news_headlines(symbol)
#         tweets = self.fetch_twitter_posts(symbol)
#         combined_text = news + tweets

#         sentiment = self.analyze_sentiment(combined_text)

#         explanation = f"Sentiment: {sentiment}, Trend: {trend}, Price: {price}"

#         if sentiment == "POSITIVE" and trend == "increasing":
#             return f"✅ BUY {symbol} — Positive sentiment and upward trend. ({explanation})"
#         elif sentiment == "NEUTRAL" and trend == "neutral":
#             return f"⚖️ HOLD {symbol} — Both sentiment and trend are neutral. ({explanation})"
#         else:
#             return f"❌ SELL {symbol} — Conditions not favorable. ({explanation})"


import requests
from config.config import config
from transformers import pipeline
from statistics import mode
import yfinance as yf

class RecommendationEngine:
    def __init__(self):
        self.news_api_key = config.NEWS_API_KEY
        self.twitter_bearer_token = config.TWITTER_BEARER_TOKEN
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    def fetch_news_headlines(self, symbol):
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.news_api_key}&language=en&pageSize=5"
        response = requests.get(url)
        if response.status_code == 200:
            return [article["title"] for article in response.json().get("articles", [])]
        return []

    def fetch_twitter_posts(self, symbol):
        query = f"{symbol} stock -is:retweet lang:en"
        url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10"
        headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return [tweet["text"] for tweet in response.json().get("data", [])]
        return []

    def analyze_sentiment(self, texts):
        if not texts:
            return "Neutral"
        results = self.sentiment_pipeline(texts)
        sentiments = [result["label"] for result in results]
        try:
            return mode(sentiments)
        except:
            return "Neutral"

    def get_price_trend(self, symbol, period="1mo", interval="1d"):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if data.empty or len(data["Close"]) < 2:
                return "neutral"
            change = data["Close"].iloc[-1] - data["Close"].iloc[0]
            if change > 0:
                return "up"
            elif change < 0:
                return "down"
            else:
                return "neutral"
        except:
            return "neutral"

    def get_market_trend(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get("sector", "").lower()

            sector_to_index = {
                "technology": "^CNXIT",         # Nifty IT
                "information technology": "^CNXIT",
                "energy": "^CNXENERGY",         # Nifty Energy
                "financial services": "^CNXFIN",  # Nifty Financial
                "healthcare": "^CNXPHARMA",     # Nifty Pharma
                "consumer defensive": "^CNXCONSUMER",  # Nifty Consumer Goods
                "utilities": "^CNX100",         # fallback
                "industrials": "^CNXIND",
                "materials": "^CNXMETAL",       # Nifty Metal
                "communication services": "^CNXMEDIA",
            }

            market_symbol = sector_to_index.get(sector, "^NSEI")  # fallback to general Nifty

            return self.get_price_trend(market_symbol)

        except Exception as e:
            print(f"[Error in get_market_trend]: {e}")
            return "neutral"


    def get_recommendation(self, symbol, market_info, sentiment=None, user_profile=None):
        price = market_info["current_price"]

        # Fetch sentiment
        news = self.fetch_news_headlines(symbol)
        tweets = self.fetch_twitter_posts(symbol)
        sentiment = self.analyze_sentiment(news + tweets)

        # Determine stock trend
        stock_trend = self.get_price_trend(symbol)
        market_trend = self.get_market_trend(symbol)

        # Make recommendation based on trend logic
        if stock_trend == "up" and market_trend == "up":
            decision = "✅ BUY"
            reason = "Stock and sector trends are positive"
        elif stock_trend == "neutral" and market_trend == "neutral":
            decision = "⚖️ HOLD"
            reason = "Both stock and sector show neutral performance"
        else:
            decision = "❌ SELL"
            reason = "Either the stock or the sector is trending downward"

        return (
            f"{decision} {symbol} — {reason}. "
            f"(Stock trend: {stock_trend}, Market trend: {market_trend})"
        )
