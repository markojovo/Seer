import time
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import math

# Load environment variables from .env file
load_dotenv()

# Alpaca API credentials
APCA_API_KEY_ID = os.getenv('ALPACA_API_KEY')
APCA_API_SECRET_KEY = os.getenv('ALPACA_SECRET_API_KEY')
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"  # for paper trading

# Initialize Alpaca API client
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')

# 1) Print buying power and each stock's weight in the portfolio
def print_portfolio():
    account = api.get_account()
    buying_power = float(account.buying_power)
    portfolio = api.list_positions()

    print("Buying power: ", buying_power)
    
    portfolio_value = float(account.portfolio_value)
    
    for position in portfolio:
        weight = (float(position.market_value) / portfolio_value) * 100
        print(f"{position.symbol}: {weight}%")

# 2) Use all of your buying power to buy a specific stock
def buy_stock_all_in(symbol):
    try:
        account = api.get_account()
        portfolio = api.list_positions()

        # Calculate total investable capital
        total_investable_capital = account.buying_power

        buying_power = math.floor(total_investable_capital)

        print(f"Buying: {symbol} in USD amount: {buying_power}")
        api.submit_order(
            symbol=symbol,
            notional=buying_power,
            side='buy',
            type='market',
            time_in_force='day'
        )
    except Exception as e:
        print(f"Error with buying stock {symbol}: {e}")

# 3) Sell all your holdings
# (This is the same function you already have)
def sell_all_positions():
    api.cancel_all_orders()
    portfolio = api.list_positions()

    for position in portfolio:
        if position.side == 'long':
            api.submit_order(
                symbol=position.symbol,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
        elif position.side == 'short':
            api.submit_order(
                symbol=position.symbol,
                qty=position.qty,
                side='buy',
                type='market',
                time_in_force='day'
            )


def buy_stocks_evenly(tickers):
    account = api.get_account()
    portfolio = api.list_positions()

    # Calculate total investable capital
    total_investable_capital = account.cash

    buying_power = float(total_investable_capital)*0.99
    # Calculate the amount of money to put into each stock
    per_stock_budget = math.floor(buying_power / len(tickers))

    # Buy each stock
    for symbol in tickers:
        try:
            print(f"Buying {symbol} for USD amount: {per_stock_budget}")
            api.submit_order(
                symbol=symbol,
                notional=per_stock_budget,
                side='buy',
                type='market',
                time_in_force='day'
            )
        except Exception as e:
            print(f"Error with buying stocks: {e}")
            continue



#So we can run alpacascripts.py and liquidate our portfolio at around 4:15 market time
if __name__ == "__main__":
    print("Selling all positions...")
    sell_all_positions()
