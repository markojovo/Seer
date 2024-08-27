import os
import csv
from dotenv import load_dotenv
import requests
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor

# Importing TotalStockList and TotalSectorList from header.py
from header import TotalStockList, TotalSectorList, news_start_date, news_end_date

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')




def fetch_news_article_urls(keyword, start_date, end_date):
    url = 'https://newsapi.org/v2/everything'
    articles_by_date = {}

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    params = {
        'q': keyword,  
        'from': start_date_str,
        'to': end_date_str,
        'sortBy': 'relevancy', #TODO: Change to relevancy then redownload the dataset, retrain
        'language': 'en',  
        'apiKey': NEWS_API_KEY
    }
    print(f"Sending News API request for keyword: {keyword}")
    response = requests.get(url, params=params)
    news_data = response.json()

    if 'articles' not in news_data:
        print(f"Error: 'articles' key not found in the News API response for keyword '{keyword}'.")
        print("Response:", news_data)

        if 'code' in news_data and news_data['code'] == 'rateLimited':  # checking for rate limited code
            raise Exception('Rate limit reached. Exiting...')  # stop the program

        return pd.DataFrame(columns=['Date', 'URLs'])

    for article in news_data['articles']:
        published_date = datetime.datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d').date()
        if published_date not in articles_by_date:
            articles_by_date[published_date] = []
        articles_by_date[published_date].append(article['url'])

    urls_df = pd.DataFrame(list(articles_by_date.items()), columns=['Date', 'URLs'])
    urls_df['Date'] = pd.to_datetime(urls_df['Date'])

    return urls_df


def get_stock_info(stock_symbol, constituents_csv_file):
    with open(constituents_csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            if row[0] == stock_symbol:
                return row[1], row[2]
    return None, None  # Return None if not found in CSV

def get_full_text_from_url(url, keyword):
    try:
        article = Article(url)
        article.download()
        article.parse()
        print(f"Success: {url}")
        full_text = f"[[[{keyword}]]] {article.text}"
        return full_text
    except Exception as e:
        print(f"Failed: {url}")
        return None


def handle_article(index, url, folder_path, keyword):
    print(f"Fetching full text for URL (as a part of {keyword}): {url}")
    full_text = get_full_text_from_url(url, keyword)
    if full_text is not None:
        # Save article to a txt file
        article_file_name = f'article{index+1}.txt'
        article_file_path = os.path.join(folder_path, article_file_name)
        with open(article_file_path, 'w',encoding="utf-8") as f:
            f.write(full_text)



if __name__ == "__main__":
    start_date = news_start_date
    end_date = news_end_date
    executor = ThreadPoolExecutor(max_workers=12)

    for stock_symbol in TotalStockList[:]:
        # Fetch company name and sector from CSV file
        company_name, sector = get_stock_info(stock_symbol, 'constituents_csv.csv')

        # If company name is not found, use just the stock symbol for the query
        if company_name is None:
            print(f"{stock_symbol} not found in the CSV file. Using stock symbol for the query.")
            keyword = f"\"{stock_symbol} stock\""
        else:
            keyword = f"\"{stock_symbol} stock\" OR \"{company_name} company\""
        
        urls_df = fetch_news_article_urls(keyword, start_date, end_date)

        # Save news data to the respective folder
        news_folder_path = os.path.join('database', stock_symbol, 'news')
        os.makedirs(news_folder_path, exist_ok=True)

        urls_folder_path = os.path.join('database', stock_symbol, 'urls')
        os.makedirs(urls_folder_path, exist_ok=True)

        for index, row in urls_df.iterrows():
            date = row['Date']
            urls = row['URLs']

            # Save URLs to a txt file
            urls_file_path = os.path.join(urls_folder_path, f'{date.strftime("%Y-%m-%d")}_urls.txt')
            with open(urls_file_path, 'w',encoding="utf-8") as f:
                for url in urls:
                    f.write(url + "\n")

            # Create a folder for this date
            date_folder_path = os.path.join(news_folder_path, date.strftime("%Y-%m-%d"))
            os.makedirs(date_folder_path, exist_ok=True)

            # Handle articles in parallel
            futures = [executor.submit(handle_article, idx, url, date_folder_path, keyword) for idx, url in enumerate(urls)]
            _ = [f.result() for f in futures]  # Ensures all tasks are completed




    # Fetch news articles for sectors
    for sector in TotalSectorList:
        keyword = sector
        urls_df = fetch_news_article_urls(keyword, start_date, end_date)

        # Save news data to the respective folder
        news_folder_path = os.path.join('database', 'sectors', sector, 'news')
        os.makedirs(news_folder_path, exist_ok=True)

        for index, row in urls_df.iterrows():
            date = row['Date']
            urls = row['URLs']

            # Save URLs to a txt file
            urls_file_path = os.path.join(news_folder_path, f'{date.strftime("%Y-%m-%d")}_urls.txt')
            with open(urls_file_path, 'w',encoding="utf-8") as f:
                for url in urls:
                    f.write(url + "\n")

            # Create a folder for this date
            date_folder_path = os.path.join(news_folder_path, date.strftime("%Y-%m-%d"))
            os.makedirs(date_folder_path, exist_ok=True)

            # This part is inside the for loop for each URL in the sector news fetching section
            for index, url in enumerate(urls):
                print(f"Fetching full text for URL (as a part of {sector} sector): {url}")
                full_text = get_full_text_from_url(url, keyword)
                if full_text is not None:
                    # Save article to a txt file
                    article_file_name = f'article{index+1}.txt'
                    article_file_path = os.path.join(date_folder_path, article_file_name)
                    with open(article_file_path, 'w',encoding="utf-8") as f:
                        f.write(full_text)


