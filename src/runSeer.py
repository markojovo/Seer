# runSeer.py

import os
import platform
   
import tensorflow as tf
if platform.system() == "Linux":
    print("Linux Detected: Running model without using GPU...")
    tf.config.set_visible_devices([], 'GPU')


from header import window_size, articles_per_day, TotalTestList, max_BERT_sequence_length, num_cols, MultiTimeDistributed, TotalStockList, TotalTradeList, max_num_cpu_threads, start_date, custom_binary_loss, ScalingLayer, custom_loss
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpacaScripts import sell_all_positions, buy_stock_all_in, buy_stocks_evenly
from modules.stock_data import fetch_stock_data
from modules.fred_data import fetch_and_merge_all_fred_data
from modules.indicators import apply_indicators, normalize_data
from updateArticleDatabase import fetch_news_article_urls, get_stock_info, get_full_text_from_url, handle_article
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.mixed_precision import set_global_policy

#set_global_policy('float16')




load_dotenv()

def retrieve_and_format_news_data(stock_symbol, start_date, end_date, articles_per_day):
    urls_df = fetch_news_article_urls(stock_symbol, start_date, end_date) # add option to look for stock name
    urls_df = urls_df.sort_values('Date')  # Ensure the dataframe is sorted by date
    urls_df.set_index('Date', inplace=True)

    # Exclude URLs from "http://markets.businessinsider.com/news/stocks/*"
    urls_df['URLs'] = urls_df['URLs'].apply(lambda urls: [url for url in urls if 'markets.businessinsider.com/news/stocks/' not in url])

    article_data = []
    for i in range((end_date - start_date).days + 1):  # For each day in the date range
        day = start_date + timedelta(days=i)
        if day in urls_df.index:
            day_urls = urls_df.loc[day, 'URLs'][:articles_per_day]  # Get the first articles_per_day URLs for this day
            
            # Retrieve the articles
            day_articles = []
            with ThreadPoolExecutor(max_workers=max_num_cpu_threads) as executor:
                futures = {executor.submit(get_full_text_from_url, url, stock_symbol): url for url in day_urls}
                for future in as_completed(futures, timeout=10):  # 10 seconds timeout
                    try:
                        result = future.result()
                        day_articles.append(result)
                    except Exception as e:
                        print(f"Error occurred: {e}")


            article_data.append(day_articles)
        else:
            article_data.append([''] * articles_per_day)  # If no articles for this day, append empty strings

    return article_data







tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model('seer_v1.h5', custom_objects={'MultiTimeDistributed': MultiTimeDistributed, 'TFBertModel': TFBertModel, 'ScalingLayer':ScalingLayer,'custom_binary_loss':custom_binary_loss, 'custom_loss':custom_loss})

numerical_end_date = pd.to_datetime(datetime.now().date()-timedelta(days=0)) #End at previous day's data (not today's)
numerical_start_date = pd.to_datetime(numerical_end_date - timedelta(days = 120)) 

filtered_end_date = numerical_end_date - timedelta(days=1)
filtered_start_date = pd.to_datetime(filtered_end_date - timedelta(days=window_size-1))



print(f"Retrieving data for period of {filtered_start_date} to {filtered_end_date}, inclusive.")

sell_all_positions()
time.sleep(5)

# Placeholder to track the best stock and its predicted growth
best_stock = None
best_stock_growth = float('-inf')

# Placeholder to track the worst stock and its predicted decline
worst_stock = None
worst_stock_decline = float('inf')

# Placeholder to track the stocks and their predicted growths
stocks_to_buy = []

# Iterate over each stock
i = 1
stockList = TotalStockList
for stock_symbol in stockList:
    print(f"Updating data for: {stock_symbol} ({i}/{len(stockList)})")
    i = i + 1
    try:
        # Fetch stock data and apply transformations
        stock_data = fetch_stock_data(stock_symbol, numerical_start_date, numerical_end_date)
        stock_data = fetch_and_merge_all_fred_data(stock_data, numerical_start_date, numerical_end_date)
        stock_data = apply_indicators(stock_data)
    except ConnectionError:
        print(f"Failed to download data for stock {stock_symbol}.")
        continue
    except ValueError:
        print(f"Improperly shaped data for stock {stock_symbol}.")
        continue
    except:
        print(f"Other error encountered when retrieving and formatting numerical data")
        continue


    days_since_start = (stock_data['Date'] - start_date).dt.days

    # Calculate the timestep position encoding
    stock_data['TimeStep'] = np.sin(2 * np.pi * (days_since_start / window_size))

    # Calculate the year position encoding
    time_period = 365
    stock_data['YearPos'] = np.sin(2 * np.pi * (days_since_start / time_period))

    # Filter stock data based on filtered dates
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    mask = (stock_data['Date'] >= filtered_start_date) & (stock_data['Date'] <= filtered_end_date)
    stock_data = stock_data.loc[mask]

    print(stock_data)
    #Apply normalization
    cols_to_normalize = [col for col in stock_data.columns if col not in ['Date','Growth']]
    stock_data[cols_to_normalize] = normalize_data(stock_data[cols_to_normalize].values)


    # Drop 'Date' column from stock_data dataframe
    stock_data = stock_data.drop(columns=['Date'])


    # Convert stock data to tensor
    stock_data_tensor = tf.expand_dims(tf.convert_to_tensor(stock_data.values), axis=0)  # Adding one dimension at the start

    '''# Retrieving and Formatting News Article Data
    try:
        company_name, sector = get_stock_info(stock_symbol, 'constituents_csv.csv')

        # If company name is not found, use just the stock symbol for the query
        if company_name is None:
            print(f"{stock_symbol} not found in the CSV file. Using stock symbol for the query.")
            keyword = f"\"{stock_symbol} stock\""
        else:
            keyword = f"\"{stock_symbol} stock\" OR \"{company_name} company\""

        article_data = retrieve_and_format_news_data(keyword, filtered_start_date, filtered_end_date, articles_per_day)
    except Exception as e:
        print(f"Error occurred while fetching news data for stock {stock_symbol}. Exception: {str(e)}")
        article_data = [[''] * articles_per_day] * ((filtered_end_date - filtered_start_date).days + 1)

    # Tokenize articles and pad to the appropriate max_BERT_sequence_length
    tokenized_articles = []
    try:
        for day_articles in article_data:
            # If day_articles is None or empty, pad with zeros
            if not day_articles:
                tokenized_articles.append(np.zeros((articles_per_day, max_BERT_sequence_length)))
            else:
                # Replace None with empty string before tokenization
                day_articles = [article if article is not None else '' for article in day_articles]

                tokenized_day_articles = [tokenizer.encode(article, max_length=max_BERT_sequence_length, truncation=True, padding='max_length') for article in day_articles]

                # Ensure the list of articles for this day contains exactly `articles_per_day` articles
                if len(tokenized_day_articles) > articles_per_day:
                    tokenized_day_articles = tokenized_day_articles[:articles_per_day]
                elif len(tokenized_day_articles) < articles_per_day:
                    padding = np.zeros((articles_per_day - len(tokenized_day_articles), max_BERT_sequence_length))
                    tokenized_day_articles = np.concatenate((tokenized_day_articles, padding))

                tokenized_articles.append(tokenized_day_articles)
    except Exception as e:
        print(f"Error during article tokenization and padding: {str(e)}")
        continue


    # Convert the tokenized articles to a tensor of the right shape
    articles_tensor = tf.expand_dims(tf.convert_to_tensor(tokenized_articles), axis=0)  # Adding one dimension at the start

    # Print out tensors
    #print("Stock data tensor:", stock_data_tensor)
    #print("Article data tensor:", articles_tensor)
    '''
    try:
        # Prepare data for model prediction
        #model_output = model.predict([articles_tensor, stock_data_tensor])
        model_output = model.predict([stock_data_tensor])
        print(f"Prediction Score: {model_output[0][0]}")
    except Exception as e:
        print(f"Error during article model prediction: {str(e)}")
        continue
    # If the output is NaN, skip this stock
    if np.isnan(model_output[0][0]):
        continue
    print()


    prediction_threshold = 0.83  # You can adjust this value as per your needs
    # If this stock's predicted growth exceeds the threshold, add it to the buy list
    if model_output[0][0] > prediction_threshold:
        stocks_to_buy.append(stock_symbol)



# Call buy_stocks_evenly function to buy the selected stocks
if stocks_to_buy:
    buy_stocks_evenly(stocks_to_buy)
else:
    print("No stocks exceeding the prediction threshold")

