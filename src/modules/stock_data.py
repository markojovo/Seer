import yfinance as yf
import pandas as pd

def fetch_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data = stock_data.reset_index()
    full_date_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['Date'])
    stock_data = pd.merge(full_date_df, stock_data, on='Date', how='left')
    stock_data = stock_data.fillna(method='ffill')
    stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    stock_data['Growth'] = (stock_data['Close'].pct_change().fillna(0))*100
    return stock_data
