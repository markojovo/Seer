import requests
import pandas as pd
import datetime
from dotenv import load_dotenv
import os

load_dotenv()
NEWS_API_KEY = os.environ['NEWS_API_KEY']

def fetch_news_articles(keyword, start_date, end_date):
    url = 'https://newsapi.org/v2/everything'
    articles_by_date = {}

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    params = {
        'q': keyword,  # can replace with f'{stock_symbol} stock'
        'from': start_date_str,
        'to': end_date_str,
        'sortBy': 'popularity',
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    news_data = response.json()

    if 'articles' not in news_data:
        print("Error: 'articles' key not found in the News API response.")
        print("Response:", news_data)
        return pd.DataFrame(columns=['Date', 'Articles', 'Article_Count'])

    for article in news_data['articles']:
        published_date = datetime.datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d').date()
        if published_date not in articles_by_date:
            articles_by_date[published_date] = []
        articles_by_date[published_date].append(article['content'])

    date_range = pd.date_range(start_date, end_date, freq='D')

    for date in date_range:
        date = date.date()
        if date not in articles_by_date:
            articles_by_date[date] = []

    news_df = pd.DataFrame(list(articles_by_date.items()), columns=['Date', 'Articles'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    news_df['Article_Count'] = news_df['Articles'].apply(len)

    return news_df
