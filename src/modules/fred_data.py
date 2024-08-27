import os
import datetime
import time
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.environ['FRED_API_KEY']
fred = Fred(api_key=FRED_API_KEY)

def fetch_and_fill_fred_data(series_code, start_date, end_date):
    before_start_date = start_date - datetime.timedelta(days=180)
    try:
        fred_data = fred.get_series(series_code, before_start_date, end_date)
    except Exception as e:
        print(f"Error fetching data for series {series_code}: {e}")
        time.sleep(5)  # If there's an error, wait 5 seconds and try again
        return fetch_and_fill_fred_data(series_code, start_date, end_date)
    fred_data = fred_data.resample('D').fillna(method='ffill').reset_index()
    fred_data = fred_data[fred_data['index'] >= start_date]
    return fred_data


def fetch_and_merge_all_fred_data(stock_data, start_date, end_date):
    m1_data = fetch_and_fill_fred_data('WM1NS', start_date, end_date)
    m1_data.columns = ['Date', 'M1_Money_Supply']

    m2_data = fetch_and_fill_fred_data('WM2NS', start_date, end_date)
    m2_data.columns = ['Date', 'M2_Money_Supply']

    cpi_data = fetch_and_fill_fred_data('CPIAUCSL', start_date, end_date)
    cpi_data.columns = ['Date', 'CPI']

    ppi_data = fetch_and_fill_fred_data('PPIACO', start_date, end_date)
    ppi_data.columns = ['Date', 'PPI']

    stock_data = stock_data.merge(m1_data, on='Date', how='left')
    stock_data = stock_data.merge(m2_data, on='Date', how='left')
    stock_data = stock_data.merge(cpi_data, on='Date', how='left')
    stock_data = stock_data.merge(ppi_data, on='Date', how='left')

    stock_data[['M1_Money_Supply', 'M2_Money_Supply', 'CPI', 'PPI']] = stock_data[['M1_Money_Supply', 'M2_Money_Supply', 'CPI', 'PPI']].fillna(method='ffill')
    stock_data[['M1_Money_Supply', 'M2_Money_Supply', 'CPI', 'PPI']] = stock_data[['M1_Money_Supply', 'M2_Money_Supply', 'CPI', 'PPI']].fillna(method='bfill')

    return stock_data
