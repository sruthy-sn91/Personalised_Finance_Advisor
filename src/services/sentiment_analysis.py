# # src/services/sentiment_analysis.py

# class SentimentAggregator:
#     def __init__(self):
#         # 
#         pass

#     def get_news_articles(self, ticker):
#         # 
#         return [
#             {"title": f"{ticker} gains momentum", "description": "The stock is performing well."},
#             {"title": f"{ticker} under pressure", "description": "There are concerns regarding growth."}
#         ]

#     def sentiment_pipeline(self, texts):
#         # Dummy sentiment analysis: if text contains 'gains', it's positive; if 'under pressure', negative.
#         results = []
#         for text in texts:
#             if "gains" in text.lower():
#                 results.append({"label": "POSITIVE", "score": 0.7})
#             elif "under pressure" in text.lower():
#                 results.append({"label": "NEGATIVE", "score": 0.7})
#             else:
#                 results.append({"label": "NEUTRAL", "score": 0.0})
#         return results

#     def get_sentiment_for_ticker(self, ticker):
#         articles = self.get_news_articles(ticker)
#         texts = [article["title"] + " " + (article.get("description") or "") for article in articles][:5]
#         if not texts:
#             texts = [f"{ticker} shows stable performance in the current market."]
#         sentiments = self.sentiment_pipeline(texts)
#         avg_score = sum(
#             res["score"] if res["label"] == "POSITIVE" else -res["score"]
#             for res in sentiments
#         ) / len(sentiments)
#         return "Positive" if avg_score > 0 else "Negative"


import os
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAggregator:
    def __init__(self):
        # Ensure the VADER lexicon is available
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()
        self.news_api_key = os.getenv("NEWSAPI_API_KEY")
        self.base_url = "https://newsapi.org/v2/everything"

    def get_news_articles(self, ticker):
        """
        Fetch news articles for a given ticker using the NewsAPI.
        Falls back to dummy data if the API key is missing or an error occurs.
        """
        if not self.news_api_key:
            print("[WARNING] NEWSAPI_API_KEY not set. Using dummy news articles.")
            return [
                {"title": f"{ticker} gains momentum", "description": "The stock is performing well."},
                {"title": f"{ticker} under pressure", "description": "There are concerns regarding growth."}
            ]

        params = {
            'q': ticker,
            'apiKey': self.news_api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 5
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "ok" and data.get("articles"):
                articles = data["articles"]
                # Return list of articles with title and description
                return [{"title": article.get("title", ""), "description": article.get("description", "")} 
                        for article in articles]
            else:
                print("[ERROR] NewsAPI did not return valid articles.")
        except Exception as e:
            print(f"[ERROR] Failed to fetch news articles for {ticker}: {e}")
        
        # Fallback to dummy news data
        return [
            {"title": f"{ticker} shows stable performance", "description": "Limited news available."}
        ]

    def sentiment_pipeline(self, texts):
        """
        Process each text using VADER's SentimentIntensityAnalyzer.
        Returns a list of dictionaries containing sentiment label and compound score.
        """
        results = []
        for text in texts:
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']
            # Use standard thresholds for sentiment classification:
            # compound >= 0.05: Positive; compound <= -0.05: Negative; otherwise Neutral.
            if compound >= 0.05:
                label = "POSITIVE"
            elif compound <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            results.append({"label": label, "score": compound})
        return results

    def get_sentiment_for_ticker(self, ticker):
        """
        For the given ticker, fetch news articles and compute an aggregated sentiment.
        Uses the compound sentiment score from VADER to decide overall sentiment.
        """
        articles = self.get_news_articles(ticker)
        texts = [f"{article.get('title', '')}. {article.get('description', '')}" for article in articles][:5]
        if not texts:
            texts = [f"{ticker} shows stable performance in the current market."]
        sentiments = self.sentiment_pipeline(texts)
        avg_compound = sum(result["score"] for result in sentiments) / len(sentiments)
        if avg_compound >= 0.05:
            return "Positive"
        elif avg_compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
