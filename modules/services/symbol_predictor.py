import nltk
import aiohttp
from bs4 import BeautifulSoup
from routes.client import HttpClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SymbolPredictor:
    def __init__(self):
        pass

    def predict(self, symbol, news_limit=5, sentiment_threshold=0.05):

        http_client = HttpClient(
            f"https://finance.yahoo.com/quote/{symbol}/news")

        response = http_client.get()

        soup = BeautifulSoup(response, "lxml")

        news_headlines = [headline.text for headline in soup.find_all("h3")[
            :news_limit]]

        # nltk.downloader.download('vader_lexicon')
        analyzer = SentimentIntensityAnalyzer()

        sentiment_scores = list(map(lambda headline: analyzer.polarity_scores(
            headline)["compound"], news_headlines))

        positive_news_count = sum(
            score > sentiment_threshold for score in sentiment_scores)
        negative_news_count = sum(
            score < -sentiment_threshold for score in sentiment_scores)

        if positive_news_count > negative_news_count:
            return 1
        elif negative_news_count > positive_news_count:
            return -1
        else:
            return 0
