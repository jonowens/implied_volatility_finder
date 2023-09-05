# Import necessary libraries
from dotenv import load_dotenv
import os
import requests
import json
import pandas as pd
import alpaca_trade_api as tradeapi


load_dotenv()

key = os.getenv('APCA_API_KEY')
secret_key = os.getenv('APCA_API_SECRET_KEY')
paper_url = os.getenv('APCA_API_PAPER_URL')
current_price = 0.00

class alpaca_trading_connection:

    def __init__(self):
        self.alpaca_api = tradeapi.REST(key, secret_key, paper_url, api_version='v2')

    def get_account_info(name):
        res = name.alpaca_api.get_account()
        return res

    def get_asset_price(name, symbol):
        data = name.alpaca_api.get_last_trade(symbol)
        return data.price

    def get_asset_history(name, symbol, timeframe, num_intervals, starting_date, ending_date):        
        
        # Format date range of historical data
        start_date = pd.Timestamp(starting_date, tz="America/New_York").isoformat()
        end_date = pd.Timestamp(ending_date, tz="America/New_York").isoformat()

        # Get historical data
        hist_data = name.alpaca_api.get_bars(symbol, timeframe, num_intervals, start=start_date, end=end_date)
        
        # Return data
        return hist_data

    def place_buy_order(name, symbol, quantity):
        order_res = name.alpaca_api.submit_order(symbol, quantity, 'buy', type='market', time_in_force='fok')
        return order_res

    def place_sell_order(name, symbol, quantity):
        order_res = name.alpaca_api.submit_order(symbol, quantity, 'sell', type='market', time_in_force='fok')
        return order_res

    def close_all_positions(name):
        close_all_pos_res = name.alpaca_api.close_all_positions()
        return close_all_pos_res

    def get_positions(name):
        pos_res = name.alpaca_api.list_positions()
        return pos_res

    # Create and submit buy order and include One Cancels Other (OTO) configuration
    def place_buy_otoco_order(name, symbol, quantity, desired_profit_percent: int=2, max_loss_percent: int=1):

        # Get current asset price
        current_price = name.get_asset_price(symbol)
        
        # Determine profit limit based on passed parameter
        profit_price = current_price * (1 + (desired_profit_percent / 100))

        # Determine loss trigger based on passed parameter
        loss_price = current_price * (1 - ((max_loss_percent) / 100))

        # Create One Cancels Other (OCO) dictionaries for bracket
        profit_dict = {
            'limit_price': profit_price
        }
        
        loss_dict = {
            'stop_price': loss_price
        }
        
        # Submit trade
        order_res = name.alpaca_api.submit_order(
            symbol,
            quantity,
            side='buy',
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            take_profit=profit_dict,
            stop_loss=loss_dict)
        
        return order_res

    def get_all_assets_to_trade(name):
        
        # Get all available assets from Alpaca
        active_assets = name.alpaca_api.list_assets(status='active')

        # Return assets
        return active_assets
