import os
import pandas as pd
import numpy as np

from modules.stock_data import fetch_stock_data
from modules.fred_data import fetch_and_merge_all_fred_data
from modules.indicators import apply_indicators, normalize_data

# Import from header.py
from header import TotalStockList as trainingStocks, start_date, end_date, filtered_start_date, filtered_end_date, window_size


def apply_cumulative_growth_label(df, n_days):
    # Convert growth percentages to scale of 1
    df['Growth'] = df['Growth'] / 100 + 1

    # Calculate the rolling product of growth for the current day and the next n-1 days
    df['Label'] = df['Growth'][::-1].rolling(window=n_days, min_periods=1).apply(np.prod, raw=True)[::-1]


    # Convert the scale back to percentages
    df['Label'] = (df['Label'] - 1) * 100
    df['Label'] = np.sign(df['Label'])

    # Convert the 'Growth' back to its original scale
    df['Growth'] = (df['Growth'] - 1) * 100

    return df




# Create the 'database' folder if it does not exist
if not os.path.exists('database'):
    os.makedirs('database')

for stock_symbol in trainingStocks:
    print("Updating data for: "+stock_symbol)
    try:
        # Fetch stock data and apply transformations
        stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
        stock_data = fetch_and_merge_all_fred_data(stock_data, start_date, end_date)
        stock_data = apply_indicators(stock_data)
        stock_data['TimeStep'] = [np.sin(2 * np.pi * i / window_size) for i in range(len(stock_data))]


        time_period = 365
        stock_data['YearPos'] = [np.sin(2 * np.pi * i / time_period) for i in range(len(stock_data))]



        #stock_data['Label'] = stock_data['Growth']

        # Apply cumulative growth label
        stock_data = apply_cumulative_growth_label(stock_data, n_days=1) # replace 3 with the actual number of days you want


        #Apply normalization
        cols_to_normalize = [col for col in stock_data.columns if col not in ['Date','Growth','Label']]
        stock_data[cols_to_normalize] = normalize_data(stock_data[cols_to_normalize].values)

        # Filter stock data based on filtered dates
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        mask = (stock_data['Date'] >= filtered_start_date) & (stock_data['Date'] <= filtered_end_date)
        stock_data = stock_data.loc[mask]

        # Create a folder for the stock symbol if it does not exist
        stock_folder = os.path.join('database', stock_symbol)
        if not os.path.exists(stock_folder):
            os.makedirs(stock_folder)

        # Save the stock data as a CSV file in the stock symbol folder
        stock_data.to_csv(os.path.join(stock_folder, f'{stock_symbol}_data.csv'), index=False)
    except Exception as e:
        print(f"Error in downloading making numerical data for: {stock_symbol}: {e}")
        continue

print("Data updated successfully.")

