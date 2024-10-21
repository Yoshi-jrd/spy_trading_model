from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime, timedelta

def get_news_sentiment(query="SPY stock market sentiment OR investor confidence", days_ago=30):
    newsapi = NewsApiClient(api_key="458f35cd1d794a4f9640174d1aa548d9")
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="relevancy"
    )
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment, articles
