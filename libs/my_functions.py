# Functions used in algorithmic trading

# Imports
from datetime import date
from email.policy import default
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import logging
import alpaca_trade_api as tradeapi
import hvplot.pandas



# Turn off warning about linking chain assignment, default='warn'
pd.options.mode.chained_assignment = None

# Create logging object
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', filename='./exceptions.log', level=logging.INFO)

# Connect to Alpaca Markets API

def initialize_alpaca_markets_connections():
    '''Initialize Alpaca Markets Connections
        Loads the security keys from .env file and creates the REST API connections to Alpaca Markets
    Returns:
        Alpca Markets REST api connection
    '''
    
    ### Load .env enviroment variables
    load_dotenv()

    # Set Alpaca API key and secret
    alpaca_api_key = os.getenv("APCA_API_KEY_ID")
    alpaca_api_secret_key = os.getenv("APCA_API_SECRET_KEY")
    alpaca_market_data_url = os.getenv("APCA_API_MARKET_DATA_URL")

    # Create the Alpaca API object
    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_api_secret_key,
        alpaca_market_data_url,
        api_version = "v2"
    )

    return api

def get_stock_data(ticker='AAPL', tf='23Hour', st='2021-01-25', et='2022-05-13'):
    '''Get Stock Data
        Makes a call to Alpaca Markets for specified symbol(s) 'open', 'high', 'low',
        'close', 'volume', 'vwap'.
    Args:
        ticker (list): List of ticker symbols to request data (can provide one symbol)
        tf (str): Timeframe for data values such as 59Min, 23Hour, 1Day, 3Month, 6Month
        st (str): Start date for data range (must be in format 'YYYY-MM-DD')
        et (str): End date for data range (must be in format 'YYYY-MM-DD')
    Returns:
        Dataframe of stock data.
    '''
    # Initialize connection to Alpaca Markets
    api = initialize_alpaca_markets_connections()

    # Request bars
    result = api.get_bars(
        symbol = ticker,
        timeframe = tf,
        start = st,
        end = et
        )
        
    result_df = result.df

    return result_df



def calculate_volatility(df, symbols, start_date, end_date):
    '''Calculate Volatility
    Args:
        df (df): Asset dataframe containing ticker symbol(s) key and
        column heading: 'close'
        symbols (list): A list of symbols to calculate.
        start_date (str): Start date for data range (must be in format 'YYYY-MM-DD')
        end_date (str): End date for data range (must be in format 'YYYY-MM-DD')
    Returns:
        Dataframe of symbols and calculated volatility (standard deviation)
    '''

    daily_returns = {}

    for item in symbols:
        # Make copy of original data to not change base dataframe and reset information for each symbol
        copied_original_df = df.copy()

        try:
            # Filter stock data for time range
            startdate_filtered_df = copied_original_df.loc[copied_original_df.index >= start_date].copy()
            start_and_enddate_filtered_df = startdate_filtered_df.loc[startdate_filtered_df.index <= end_date].copy()
        except Exception as e:
            # Log exceptions and continue to next symbol
            logger.info(f'\nIssue that was thrown is related to calculating volatility:\n{e}\nSymbol is:\n{item}\nDataframe is: start_and_enddate_filtered_df\n{start_and_enddate_filtered_df}')
            continue
            
        # Filter stock data for each symbol
        start_end_symbol_result_df = start_and_enddate_filtered_df[start_and_enddate_filtered_df['symbol'] == item].copy()
        
        start_end_symbol_result_df = pd.DataFrame(start_end_symbol_result_df)

        # Daily return
        daily_returns[item] = start_end_symbol_result_df['close'].dropna().pct_change()

    daily_returns = pd.DataFrame(daily_returns)
    
    # Volatility (standard deviation)
    daily_std = daily_returns.std()

    # Annualize standard deviation
    annualized_std = daily_std * np.sqrt(252)

    # Sort the values
    annualized_std_sorted = annualized_std.sort_values()

    # Drop na
    annualized_std_sorted_no_na = annualized_std_sorted.dropna()

    return annualized_std_sorted_no_na

def rsi_breakin(data, period=14.0, overbought=70.0, oversold=30.0):
    '''
    'Relative Strength Index Beakin Strategy'
    -Takes in a dataframe with at least one column 'close' and a datetime index
        *Optionally, it can take exponential moving average window ("period"; default = 14)
        *Optionally, it can take overbought value ("overbought"; default = 70)
        *Optionally, it can take oversold value ("oversold"; default = 30)
    -Returns a dataframe with 'close', 'rsi' and 'signal'
    '''

    # But experts believe that the best timeframe for RSI actually lies between 2 to 6.
    # Intermediate and expert day traders prefer the latter timeframe as they can decrease
    # or increase the values according to their position.  Default is 14.
    
    # RSI indicator formula:
    # RS = RS_gain / abs(RS_loss)
    # RSI = 100 - [100/(1 + RS)]
    # RS_gain = {[(average gain from previous period - 1) * 13] + current gain} / 14
    # RS_loss = {[(average loss from previous period - 1) * 13] + current loss} / 14
    # Default period = 14

    # add a few new columns to the data frame to track the RSI parameters:
    data['close_chg'] = data['close'].diff()

    data['gain'] = data.close_chg.apply(lambda x: x if x >= 0 else 0)
    data['avg_gain'] = data.gain.ewm(span=period).mean()
    data['prev_avg_gain'] = data.avg_gain.shift()
    data['rs_gain'] = ((data.prev_avg_gain * (period - 1)) + data.gain) / period

    data['loss'] = abs(data.close_chg.apply(lambda x: x if x < 0 else 0))
    data['avg_loss'] = data.loss.ewm(span=period).mean()
    data['prev_avg_loss'] = data.avg_loss.shift()
    data['rs_loss'] = ((data.prev_avg_loss * (period - 1)) + data.loss) / period

    data['rs'] = data.rs_gain / data.rs_loss
    data['rsi'] = 100 - (100 / (1 + data.rs))

    data = data[['close', 'rsi']]

    # create a sub-function that will determine if the RSI is overbought, oversold or neutral
    def rsi_level(x):
        if x >= overbought:
            return -1
        elif overbought > x > oversold:
            return 1
        elif x <= oversold:
            return 0

    # add a signal column that will help identify the most basic trend
    # using the rsi_level function above, add a new column to the dataframe that returns the signal
    data['signal'] = data.rsi.apply(lambda x: rsi_level(x))

    # -1 means the RSI is over the overbought value and considered bearish potential (default = 70)
    # 0 means the RSI is between the overbought and the oversold values and considered neutral
    # 1 means the RSI is below the oversold value and considered bullish potential (defualt = 30)
    data['entry/exit'] = data['signal'].diff()

    # Initialize variable to not have multiple buys or multiple sells back to back
    bought = False
    midpoint = 0.0

    # Determines midpoint between overbought and oversold to compare rsi values
    midpoint = (overbought + oversold) / 2

    for index, row in data.iterrows():
        # Check for rsi signals moving lower than oversold line
        if bought == False and row['signal'] == 0.0 and row['entry/exit'] == -1:
            bought = True
            data.loc[index, 'holding assets'] = bought
        elif bought == True and row['signal'] == 0.0 and row['entry/exit'] == -1:
            data.loc[index, 'holding assets'] = bought

        # Check for rsi signals moving higher than oversold line
        elif bought == False and row['signal'] == 1.0 and row['entry/exit'] == 1.0:
            row['entry/exit'] = 0.0
        elif bought == True and row['signal'] == 1.0 and row['entry/exit'] == 1.0:
            row['entry/exit'] = 0.0
        # Check for rsi signal moving higher than the oversold line in one time interval
        # (Example: RSI changed from below oversold one day and above overbought the next day)
        elif bought == False and row['signal'] == -1.0 and row['entry/exit'] == -1.0:
            row['entry/exit'] = 0.0
        elif bought == True and row['signal'] == -1.0 and row['entry/exit'] == -1.0:
            bought = False
            data.loc[index, 'holding assets'] = bought

        # Check for other rsi signals moving higher than the oversold line
        elif bought == False and row['signal'] == -1.0 and row['entry/exit'] == -2.0:
            row['entry/exit'] = 0.0
        elif bought == True and row['signal'] == -1.0 and row['entry/exit'] == -2.0:
            bought = False
            data.loc[index, 'holding assets'] = bought

        # Check for rsi signals moving lower than overbought line
        elif bought == False and row['signal'] == 1.0 and row['entry/exit'] == 2.0:
            row['entry/exit'] = 0.0
        # Check for rsi signal moving lower than overbought line in one time interval
        # (Example: RSI changed from above overbought one day and below oversold the next day)
        elif row['signal'] == 0.0 and row['entry/exit'] == 1:
            bought = True
            data.loc[index, 'holding assets'] = bought

    data['overbought'] = overbought
    data['middle'] = midpoint
    data['oversold'] = oversold

    return data

def rsi_breakout(data, period=14.0, overbought=70.0, oversold=30.0):
    '''
    'Relative Strength Index Beakout'
    -Takes in a dataframe with at least one column 'close' and a datetime index
        *Optionally, it can take exponential moving average window ("period"; default = 14)
        *Optionally, it can take overbought value ("overbought"; default = 70)
        *Optionally, it can take oversold value ("oversold"; default = 30)
    -Returns a dataframe with 'close', 'rsi' and 'signal'
    '''

    # But experts believe that the best timeframe for RSI actually lies between 2 to 6.
    # Intermediate and expert day traders prefer the latter timeframe as they can decrease
    # or increase the values according to their position.  Default is 14.
    
    # RSI indicator formula:
    # RS = RS_gain / abs(RS_loss)
    # RSI = 100 - [100/(1 + RS)]
    # RS_gain = {[(average gain from previous period - 1) * 13] + current gain} / 14
    # RS_loss = {[(average loss from previous period - 1) * 13] + current loss} / 14
    # Default period = 14

    # add a few new columns to the data frame to track the RSI parameters:
    data['close_chg'] = data['close'].diff()

    data['gain'] = data.close_chg.apply(lambda x: x if x >= 0 else 0)
    data['avg_gain'] = data.gain.ewm(span=period).mean()
    data['prev_avg_gain'] = data.avg_gain.shift()
    data['rs_gain'] = ((data.prev_avg_gain * (period - 1)) + data.gain) / period

    data['loss'] = abs(data.close_chg.apply(lambda x: x if x < 0 else 0))
    data['avg_loss'] = data.loss.ewm(span=period).mean()
    data['prev_avg_loss'] = data.avg_loss.shift()
    data['rs_loss'] = ((data.prev_avg_loss * (period - 1)) + data.loss) / period

    data['rs'] = data.rs_gain / data.rs_loss
    data['rsi'] = 100 - (100 / (1 + data.rs))

    data = data[['close', 'rsi']]

    # create a sub-function that will determine if the RSI is overbought, oversold or neutral
    def rsi_level(x):
        if x >= overbought:
            return -1
        elif overbought > x > oversold:
            return 1
        elif x <= oversold:
            return 0

    # add a signal column that will help identify the most basic trend
    # using the rsi_level function above, add a new column to the dataframe that returns the signal
    data['signal'] = data.rsi.apply(lambda x: rsi_level(x))

    # -1 means the RSI is over the overbought value and considered bearish potential (default = 70)
    # 0 means the RSI is between the overbought and the oversold values and considered neutral
    # 1 means the RSI is below the oversold value and considered bullish potential (defualt = 30)
    data['entry/exit'] = data['signal'].diff()

    # Initialize variable to not have multiple buys or multiple sells back to back
    bought = False
    midpoint = 0.0

    # Determines midpoint between overbought and oversold to compare rsi values
    midpoint = (overbought + oversold) / 2

    for index, row in data.iterrows():
        # Check for rsi signals moving lower than oversold line
        if (row['signal'] == 0.0 and row['entry/exit'] == -1.0):
            row['entry/exit'] = 0.0
        # Check for rsi signal moving lower than oversold line in one time interval
        # (Example: RSI changed from above overbought one day and below oversold the next day)
        elif bought == False and row['signal'] == 0.0 and row['entry/exit'] == 1.0:
            # Do not signal to buy when dropping below oversold line
            row['entry/exit'] = 0.0
        elif bought == True and row['signal'] == 0.0 and row['entry/exit'] == 1.0:
            bought = False
            data.loc[index, 'holding assets'] = bought


        # Check for rsi signals moving higher than oversold line
        elif bought == False and row['signal'] == 1.0 and row['entry/exit'] == 1.0:
            bought = True
            data.loc[index, 'holding assets'] = bought
        elif bought == True and row['signal'] == 1.0 and row['entry/exit'] == 1.0:
            data.loc[index, 'holding assets'] = bought
        # Check for rsi signal moving higher than the oversold line in one time interval
        # (Example: RSI changed from below oversold one day and above overbought the next day)
        elif bought == False and row['signal'] == -1.0 and row['entry/exit'] == -1.0:
            bought = True
            data.loc[index, 'holding assets'] = bought
        elif bought == True and row['signal'] == -1.0 and row['entry/exit'] == -1.0:
            data.loc[index, 'holding assets'] = bought


        # Check for rsi signals moving lower than overbought line
        elif bought == False and row['signal'] == 1.0 and row['entry/exit'] == 2.0:
            # Do not signal to sell when dropping below oversold line and nothing has been previously bought
            row['entry/exit'] = 0.0
        elif bought == True and row['signal'] == 1.0 and row['entry/exit'] == 2.0:
            bought = False
            data.loc[index, 'holding assets'] = bought

    data['overbought'] = overbought
    data['middle'] = midpoint
    data['oversold'] = oversold

    return data

def configure_macd_and_ewma(data, period_fast=12, period_slow=26, period_signal=9):

    '''
    'Moving Average Convergence/Divergence'
    -Takes in a dataframe with at least one column 'close' and a datetime as the index
        *Optionally, takes the slow ewma window size ("period_slow"; default = 26)
        *Optionally, takes the fast ewma window size ("period_fast"; default = 12)
        *Optionally, takes signal line window size ("period_signal"; default = 9)
    -Returns a dataframe with 'close', 'slow_ewma', 'fast_ewma', 'macd', 'signal_line', 'con_div', 'macd_signal', 'condiv_signal', and 'signal'
    '''
    # MACD(5,35,5) is more sensitive than MACD(12,26,9) and might be better suited for weekly charts.
    # The Moving Average Convergence/Divergence (MACD) indicator can be broken into three parts: the signal line, the MACD line, and the convergence/divergence between the two
    # THE MACD LINE
    # The MACD line is created by subtracting a slow EMA from a fast EMA.  
        # The defualts are: 26 for slow and 12 for fast.  Adjust to suit your fancy
    slow_ewma = data.close.ewm(span=period_slow).mean()

    fast_ewma = data.close.ewm(span=period_fast).mean()

    macd = fast_ewma - slow_ewma

    # THE SIGNAL LINE
    # The signal line is generated by taking an EWMA of the MACD line.  
        # The default EWMA to use is 9 periods, but adjust as you see fit
    signal_line = macd.ewm(span=period_signal).mean()

    # THE CONVERGENCE/DIVERGENCE
    # The convergence/divergence of the MACD and signal lines is simply the MACD minus the signal.  
        # When it is below zero, it is considered bearish and when it is above zero it is considered bullish
    condiv = macd - signal_line

    # build out a dataframe that houses all the MACD signal data
    macd_df = pd.DataFrame(
        {'close': data.close,
        'slow_ewma': slow_ewma,
        'fast_ewma': fast_ewma,
        'macd': macd,
        'signal_line': signal_line,
        'con_div': condiv}
        )

    # add a column that returns a 0 or 1 for the MACD signal.  
        # -1 means the MACD is below the zero line and is bearish.  
        # 1 means the MACD is above the zero line and is bullish.
    macd_df['macd_signal'] = macd_df['macd'].apply(lambda x: 1 if x > 0 else -1)

    # add a column that return a 0 or 1 for the convergence/divergence.  
        # -1 means it is negative and bearish.  
        # 1 means it is positive and bullish. 
    macd_df['condiv_signal'] = macd_df['con_div'].apply(lambda x: 1 if x > 0 else -1)

    # add a column that returns a -1, 0, or 1.  This column is the sum of the previous two.  
        # -1 is bearish (all signals are bearish)
        # 0 is neutral (one signal is bullish and one is bearish) 
        # 1 is bullish (all signals are bullish)
    macd_df['signal'] = (macd_df.macd_signal + macd_df.condiv_signal)/2

    return macd_df

def create_summary_data_stocks(dataframe, capital=100000.00, num_share_size=500):
    '''Create Summary For Data (Stocks)
    Function is used for summary analysis of single entry and single exit stock strategies.
    Args:
        dataframe (df): Contains initial dataframe of information to summarize
        capital (fl): Initial account value
        num_share_size (dec): Size of position to take when strategy is triggered
    Returns:
        Dataframe of Portfolio Holdings', 'Portfolio Cash', 'Portfolio PnL', 'Portfolio Total',
        'Portfolio Daily Returns', and 'Portfolio Cumulative Returns'.
    '''

    # Set initializations
    initial_capital = float(capital)
    cumulate_buy_stock_purchases = 0
    cumulate_close_values = 0.00
    average_close_values = 0.00

    # Set the share size
    share_size = num_share_size

    # Take a 500 share position where RSI exits above 30
    # Release a 500 share position where RSI exits below 70
    for index, row in dataframe.iterrows():
        if row['signal'] == 1.0 and row['entry/exit'] == 1.0:
                dataframe.loc[index, 'Entry/Exit Position'] = 'Buy'
                cumulate_buy_stock_purchases += 1
                cumulate_close_values += row['close']
                dataframe.loc[index, 'Stock Value'] = row['close']
                dataframe.loc[index, 'Position'] = share_size
                continue
        if row['signal'] == 1.0 and row['entry/exit'] == 2.0:
            average_close_values = cumulate_close_values / cumulate_buy_stock_purchases
            if average_close_values < row['close']:
                dataframe.loc[index, 'Entry/Exit Position'] = 'Sell'
                dataframe.loc[index, 'Stock Value'] = average_close_values
                dataframe.loc[index, 'Position'] = share_size * cumulate_buy_stock_purchases
                cumulate_buy_stock_purchases = 0
                cumulate_close_values = 0.00
                average_close_values = 0.00
                continue
            else:
                dataframe.loc[index, 'Entry/Exit Position'] = 'Lost Money/Close Positions'
                dataframe.loc[index, 'Stock Value'] = 0.00
                dataframe.loc[index, 'Position'] = 0
                cumulate_buy_stock_purchases = 0
                cumulate_close_values = 0.00
                average_close_values = 0.00
                continue
        else:
            dataframe.loc[index, 'Stock Value'] = 0.00
            dataframe.loc[index, 'Position'] = 0

    # Multiply share price by entry/exit positions and get the cumulatively sum
    dataframe['Portfolio Holdings'] = dataframe['close'] * dataframe['Position']

    # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
    dataframe['Portfolio Cash'] = initial_capital - dataframe['Portfolio Holdings']

    # Subtract the Stock Value (averages of purchases) from the current close and multiple by the shares per contract for the profit and loss
    dataframe['Portfolio PnL'] = (dataframe['close'] - dataframe['Stock Value']) * dataframe['Position']

    # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
    dataframe['Portfolio Total'] = dataframe['Portfolio Cash'] + (dataframe['Portfolio PnL']).cumsum()

    # Calculate the portfolio daily returns
    dataframe['Portfolio Daily Returns'] = dataframe['Portfolio Total'].pct_change()

    # Calculate the cumulative returns
    dataframe['Portfolio Cumulative Returns'] = ((1 + dataframe['Portfolio Daily Returns']).cumprod() - 1) * 100

    # Return dataframe information
    return dataframe


def backtest_stock_data(
    account_value = 100000,
    share_size = 100,
    min_allowable_share_value = 0.0,
    max_allowable_share_value = 1000.00,
    ewma_period = 14,
    overbought_value = 70.0,
    oversold_value = 30.0,
    startdate_range = '2021-01-03',
    enddate_range = '2022-06-10',
    companies_list = 'AMZN'
    ):
    '''Backtest Stock Data
    Backtesting function for stock data using single entry and single exit RSI strategy
    Args:
        account_value (fl): Initial size of account in dollars
        share_size (dec): Size of position to take when triggered
        min_allowable_share_value (fl): Minimum close value to trade
        max_allowable_share_value (fl): Maximum close value to trade
        ewma_period (dec): Exponential weighted moving average window (usually 14)
        overbought_value (fl): Overbought value (usually 70)
        oversold_value (fl): Oversold value (usually 30)
        startdate_range (date): Starting date in range to backtest
        enddate_range (date): Ending date in range to backtest
        companies_list (list): List of ticker symbols analyze
    Returns:
        Lowest, highest, and average stock prices along with account value at end
        of simulation.
    '''

    total_profit = 0.0
    pnl = 0.0
    num_companies = 0
    running_stock_price = 0.0
    avg_stock_price = 0.0
    lowest_stock_price = 9999999.0
    lowest_stock_price_company = ''
    highest_stock_price = 0.0
    highest_stock_price_company = ''

    # Get and analyze stock data for each symbol in sp500 one symbol at a time
    for item in companies_list:
        # Count companies
        num_companies += 1

        # Get stock data from Alpaca Markets
        result_df = get_stock_data(item, '23Hour', startdate_range, enddate_range)
        
        # Try to read data for current symbol
        try:
            if min_allowable_share_value < result_df['close'][-1] and result_df['close'][-1] <= max_allowable_share_value:
                # Create RSI data for symbol
                signals_df = rsi_breakout(result_df, ewma_period, overbought_value, oversold_value)
            else:
                continue
        # Track exceptions and continue to next symbol
        except Exception as e:
            # Log Exception Data
            logger.info(f'\nException thrown:\n{e}\nSymbol is:\n{item}\nDataframe is:\n{result_df}')
            continue
        
        # Create summary data for symbol
        summary_results = create_summary_data_stocks(signals_df, account_value, share_size)

        # Determine if current symbol is still held in account to calculate profit and loss is sold at current end time closing cost
        if summary_results['holding assets'][-1] == True:
            # If symbol is currently held, calculate profit and loss if sold at current close value
            #pnl = ((summary_results['close'][-1] * num_contracts) + summary_results['Portfolio Cash'][-1]) - account_value
            
            # Calculate current profit and loss accumulated over time
            pnl = summary_results['Portfolio Total'][-1] - account_value
            
            # Add profit and loss to running total for all entire account
            total_profit += pnl

            # If symbol is currently held, can continue and not calculate profit and loss for those stocks until sold
            print('{:3d} - Symbol: {:<6s}    Close: {:11.2f}    PnL: {:11.2f}    Cumulative Total Profit: {:11.2f}    Holding: {!s}'.format(num_companies, item, summary_results['close'][-1], pnl, total_profit, summary_results['holding assets'][-1]))

        else:
            # If symbol is NOT currently held in account, calculate profit and loss
            pnl = summary_results['Portfolio Total'][-1] - account_value

            # Add profit and loss to running total for all entire account
            total_profit += pnl

            # Print summary data for symbol being analyzed
            print('{:3d} - Symbol: {:<6s}    Close: {:11.2f}    PnL: {:11.2f}    Cumulative Total Profit: {:11.2f}    Holding: {!s}'.format(num_companies, item, summary_results['close'][-1], pnl, total_profit, summary_results['holding assets'][-1]))

        # Calculate average stock price of all sp500 close values
        running_stock_price += summary_results['close'][-1]
        avg_stock_price = running_stock_price / num_companies

        # Determines current lowest stock price out of sp500 companies
        if summary_results['close'][-1] <= lowest_stock_price:
            lowest_stock_price = summary_results['close'][-1]
            lowest_stock_price_company = item

        # Determines current highest stock price out of sp500 companies
        if summary_results['close'][-1] >= highest_stock_price:
            highest_stock_price = summary_results['close'][-1]
            highest_stock_price_company = item

    print('Current Lowest Stock Price: {:11.2f}, Symbol: {:>6s}'.format(lowest_stock_price, lowest_stock_price_company))
    print('Current Highest Stock Price: {:11.2f}, Symbol: {:>6s}'.format(highest_stock_price, highest_stock_price_company))
    print('Current Average Stock Price: {:11.2f}'.format(avg_stock_price))

def create_summary_data_options(dataframe, initial_capital=100000.00, num_contracts=1):
    # Initiate variables
    cumulate_buy_call_options = 0
    cumulate_close_values = 0.00
    
    num_shares = 100 * num_contracts

    # Determine when to Buy Call Option Contract and when contract is exercised or allowed to expire
    for index, row in dataframe.iterrows():
            if row['signal'] == 1.0 and row['entry/exit'] == 1.0:
                dataframe.loc[index, 'Entry/Exit Position'] = 'Buy Call Option'
                cumulate_buy_call_options += 1
                cumulate_close_values += row['close']
                dataframe.loc[index, 'Call Value'] = 0.00
                dataframe.loc[index, 'Position'] = 0
                continue
            if row['signal'] == 1.0 and row['entry/exit'] == 2.0:
                average_close_values = cumulate_close_values / cumulate_buy_call_options
                if average_close_values < row['close']:
                    dataframe.loc[index, 'Entry/Exit Position'] = 'Option Exercised'
                    dataframe.loc[index, 'Call Value'] = average_close_values
                    dataframe.loc[index, 'Position'] = num_shares * cumulate_buy_call_options
                    cumulate_buy_call_options = 0
                    cumulate_close_values = 0.00
                    average_close_values = 0.00
                    continue
                else:
                    dataframe.loc[index, 'Entry/Exit Position'] = 'Option Expired'
                    dataframe.loc[index, 'Call Value'] = 0.00
                    dataframe.loc[index, 'Position'] = 0
                    cumulate_buy_call_options = 0
                    cumulate_close_values = 0.00
                    average_close_values = 0.00
                    continue
            else:
                dataframe.loc[index, 'Call Value'] = 0.00
                dataframe.loc[index, 'Position'] = 0

    # Multiply share price by entry/exit positions and get the cumulatively sum
    dataframe['Portfolio Holdings'] = dataframe['Call Value'] * dataframe['Position']

    # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
    dataframe['Portfolio Cash'] = initial_capital - dataframe['Portfolio Holdings']

    # Subtract the option contract close from the current close and multiple by the shares per contract for the profit and loss
    dataframe['Portfolio PnL'] = (dataframe['close'] - dataframe['Call Value']) * dataframe['Position']

    # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
    dataframe['Portfolio Total'] = dataframe['Portfolio Cash'] + (dataframe['Portfolio PnL']).cumsum()

    # Calculate the portfolio daily returns
    dataframe['Portfolio Daily Returns'] = dataframe['Portfolio Total'].pct_change()

    # Calculate the cumulative returns
    dataframe['Portfolio Cumulative Returns'] = ((1 + dataframe['Portfolio Daily Returns']).cumprod() - 1) * 100

    # Return dataframe information
    return dataframe

def backtest_option_data(
    account_value=100000.00,
    num_contracts=1,
    min_allowable_share_value = 0.0,
    max_allowable_share_value = 1000.00,
    ewma_period = 14,
    overbought_value = 70.0,
    oversold_value = 30.0,
    startdate_range = '2022-01-03',
    enddate_range = '2022-06-23',
    companies_list = 'AMZN'
    ):
    '''Backtest Option Data
    Backtesting function for option contract data using multiple entry and single exit RSI strategy
    Args:
        account_value (fl): Initial size of account in dollars
        num_contracts (dec): Size of position to take when triggered
        min_allowable_share_value (fl): Minimum close value to trade
        max_allowable_share_value (fl): Maximum close value to trade
        ewma_period (dec): Exponential weighted moving average window (usually 14)
        overbought_value (fl): Overbought value (usually 70)
        oversold_value (fl): Oversold value (usually 30)
        startdate_range (date): Starting date in range to backtest
        enddate_range (date): Ending date in range to backtest
        companies_list (list): List of ticker symbols analyze
    Returns:
        Lowest, highest, and average stock prices along with account value at end
        of simulation.
    '''

    total_profit = 0.0
    pnl = 0.0
    num_companies = 0
    running_stock_price = 0.0
    avg_stock_price = 0.0
    lowest_stock_price = 9999999.0
    lowest_stock_price_company = ''
    highest_stock_price = 0.0
    highest_stock_price_company = ''


    # Get and analyze stock data for each symbol in sp500 one symbol at a time
    for item in companies_list:
        # Count companies
        num_companies += 1

        # Get stock data from Alpaca Markets
        result_df = get_stock_data(item, '23Hour', startdate_range, enddate_range)
        
        # Try to read data for current symbol
        try:
            if min_allowable_share_value < result_df['close'][-1] and result_df['close'][-1] <= max_allowable_share_value:
                # Create RSI data for symbol
                signals_df = rsi_breakout(result_df, ewma_period, overbought_value, oversold_value)
            else:
                continue
        # Track exceptions and continue to next symbol
        except Exception as e:
            # Log Exception Data
            logger.info(f'\nException thrown:\n{e}\nSymbol is:\n{item}\nDataframe is:\n{result_df}')
            continue
        
        # Create summary data for symbol
        summary_results = create_summary_data_options(signals_df, account_value, num_contracts)

        # Determine if current symbol is still held in account to calculate profit and loss is sold at current end time closing cost
        if summary_results['holding assets'][-1] == True:
            # If symbol is currently held, calculate profit and loss if sold at current close value
            #pnl = ((summary_results['close'][-1] * num_contracts) + summary_results['Portfolio Cash'][-1]) - account_value
            
            # Calculate current profit and loss accumulated over time
            pnl = summary_results['Portfolio Total'][-1] - account_value
            
            # Add profit and loss to running total for all entire account
            total_profit += pnl

            # If symbol is currently held, can continue and not calculate profit and loss for those options still active until sold
            print('{:3d} - Symbol: {:<6s}    Close: {:11.2f}    PnL: {:11.2f}    Cumulative Total Profit: {:11.2f}    Holding: {!s}'.format(num_companies, item, summary_results['close'][-1], pnl, total_profit, summary_results['holding assets'][-1]))
    
        else:
            # If symbol is NOT currently held in account, calculate profit and loss
            pnl = summary_results['Portfolio Total'][-1] - account_value
        
            # Add profit and loss to running total for all entire account
            total_profit += pnl
            
            # Print summary data for symbol being analyzed
            print('{:3d} - Symbol: {:<6s}    Close: {:11.2f}    PnL: {:11.2f}    Cumulative Total Profit: {:11.2f}    Holding: {!s}'.format(num_companies, item, summary_results['close'][-1], pnl, total_profit, summary_results['holding assets'][-1]))

        # Calculate average stock price of all sp500 close values
        running_stock_price += summary_results['close'][-1]
        avg_stock_price = running_stock_price / num_companies

        # Determines current lowest stock price out of companies list
        if summary_results['close'][-1] <= lowest_stock_price:
            lowest_stock_price = summary_results['close'][-1]
            lowest_stock_price_company = item

        # Determines current highest stock price out of companies list
        if summary_results['close'][-1] >= highest_stock_price:
            highest_stock_price = summary_results['close'][-1]
            highest_stock_price_company = item

    print('Current Lowest Stock Price: {:11.2f}, Symbol: {:>6s}'.format(lowest_stock_price, lowest_stock_price_company))
    print('Current Highest Stock Price: {:11.2f}, Symbol: {:>6s}'.format(highest_stock_price, highest_stock_price_company))
    print('Current Average Stock Price: {:11.2f}'.format(avg_stock_price))


def plot_ewma(
    ewma_df, 
    company_symbol='AMZN', 
    output='screen', 
    file_name='sample', 
    window_width=1000, 
    window_height=400, 
    chart_key_position='top_left'
    ):
    '''Plot exponential weighted moving average lines
        Plots EWMA lines along with allowing for various outputs.
    Args:
        ewma_df (dataframe): Dataframe containing 'close' values for single stocks
        output (string): 'screen' allows for display plot to screen and 'file' saves to file.
        file_name (string): Name of .png image to save
        window_width (Int): Width of window to size the plot window
        window_height (Int): Height of window to size the plot window
        chart_key_position (string): 'top_left', 'bottom_left', 'top_right', 'bottom_right'
    Outputs:
        To screen EWMA plots or saves plot to file.
    '''

    # Visualize slow and fast ewma
    slow = ewma_df[['slow_ewma']].hvplot.line(
        line_color='green',
        ylabel='Price in $',
        width=window_width,
        height=window_height,
        label='Slow'
    )

    fast = ewma_df[['fast_ewma']].hvplot.line(
        line_color='red',
        ylabel='Price in $',
        width=window_width,
        height=window_height,
        label='Fast'
    )

    # Visualize close price for the investment
    security_close = ewma_df[['close']].hvplot.line(
        title= f'EWMA    Symbol: {company_symbol}',
        line_color='blue',
        ylabel='Price in $',
        width=window_width,
        height=window_height,
        label='Stock Close'
    )

    # Displays the slow and fast exponential weighted moving average plots
    slow_fast_ewma_plot = slow * fast * security_close

    if output == 'file':
        
        # Save plot charts
        hvplot.save(slow_fast_ewma_plot, f'../../../Desktop/new/{file_name}.html')
    elif output == 'screen':

        # Return plot charts
        return slow_fast_ewma_plot.opts(xaxis=None, show_legend=True, legend_position=chart_key_position)

def plot_macd(
    macd_df, 
    company_symbol='AMZN', 
    output='screen', 
    file_name='sample', 
    window_width=1000, 
    window_height=400, 
    chart_key_position='top_left'
    ):
    '''Plot MACD
        Plots MACD, signal, and convergence divergence lines along with allowing for
            various outputs.
    Args:
        macd_df (dataframe): Dataframe containing 'close' values for single stocks
        output (string): 'screen' allows for display plot to screen and 'file' saves to file.
        file_name (string): Name of .png image to save
        window_width (Int): Width of window to size the plot window
        window_height (Int): Height of window to size the plot window
        chart_key_position (string): 'top_left', 'bottom_left', 'top_right', 'bottom_right'
    Outputs:
        To screen MACD plots or saves plot to file.
    '''
    
    # Visualize macd line
    macd = macd_df[['macd']].hvplot.line(
        ylabel='Price',
        color='blue',
        width=window_width,
        height=window_height,
        label='MACD'
    )

    # Visulatize signal line
    signal_line = macd_df[['signal_line']].hvplot.line(
        ylabel='Price',
        color='yellow',
        width=window_width,
        height=window_height,
        label='Signal'
    )

    # Visualize convergence divergence line
    con_div = macd_df[['con_div']].hvplot.line(
        title= f'MACD    Symbol: {company_symbol}',
        ylabel='Price',
        color='purple',
        width=window_width,
        height=window_height,
        label='Convergence Divergence'
    )

    # Overlay plots
    macd_stuff_plot = macd * signal_line * con_div

    if output == 'file':
        
        # Save plot charts
        hvplot.save(macd_stuff_plot, f'../../../Desktop/new/{file_name}.html')
    elif output == 'screen':

        # Return plot charts
        return macd_stuff_plot.opts(xaxis=None, show_legend=True, legend_position=chart_key_position)

def plot_rsi_entry_exit_points(
    entry_exit_df, 
    company_symbol='AMZN', 
    output='screen', 
    file_name='sample', 
    window_width=1000, 
    window_height=400, 
    chart_key_position='top_left'
    ):
    '''Plot RSI Entry Exit Points
        Plots entry and exit points for custom RSI strategy and allows for
            various outputs.
    Args:
        signals_df (dataframe): Dataframe containing 'entry/exit' points (1 and 2)
            along with 'close' values for single stocks
        output (string): 'screen' allows for display plot to screen and 'file' saves to file.
        file_name (string): Name of .png image to save
        window_width (Int): Width of window to size the plot window
        window_height (Int): Height of window to size the plot window
        chart_key_position (string): 'top_left', 'bottom_left', 'top_right', 'bottom_right'
    Outputs:
        To screen entry and exit plot indicators or saves plot to file.
    '''
    # Visualize entry position relative to close price
    entry = entry_exit_df[entry_exit_df['holding assets'] == True]['close'].hvplot.scatter(
        color='green',
        ylabel='Price in $',
        width=window_width,
        height=window_height,
        label='Entry'
    )

    # Visualize exit position relative to close price
    exit = entry_exit_df[entry_exit_df['holding assets'] == False]['close'].hvplot.scatter(
        color='red',
        ylabel='Price in $',
        width=window_width,
        height=window_height,
        label='Exit'
    )

    # Visualize close price for the investment
    security_close = entry_exit_df[['close']].hvplot.line(
        title= f'RSI Entry/Exit Points    Symbol: {company_symbol}',
        line_color='lightgray',
        ylabel='Price in $',
        width=window_width,
        height=window_height,
        label='Stock Close'
    )

    # Overlay plots
    entry_exit_plot = security_close * entry * exit

    if output == 'file':
        
        # Save plot charts
        hvplot.save(entry_exit_plot, f'../../../Desktop/new/{file_name}.html')
    elif output == 'screen':

        # Return plot charts
        return entry_exit_plot.opts(xaxis=None, show_legend=True, legend_position=chart_key_position)


def plot_raw_data_for_rsi(
        df, 
        company_symbol='AMZN', 
        output='screen', 
        file_name='relative_strength_sample', 
        window_width=1000, 
        window_height=400, 
        chart_key_position='top_left'
        ):
    '''Plot RSI Raw Data
        Plots overbought, middle, oversold and RSI values for custom RSI strategy and allows for
            various outputs.
    Args:
        df (dataframe): Dataframe containing 'entry/exit' points (1 and 2)
            along with 'close' values for single stocks
        output (string): 'screen' allows for display plot to screen and 'file' saves to file.
        file_name (string): Name of .png image to save
        window_width (Int): Width of window to size the plot window
        window_height (Int): Height of window to size the plot window
        chart_key_position (string): 'top_left', 'bottom_left', 'top_right', 'bottom_right'
    Outputs:
        To screen entry and exit plot indicators or saves plot to file.
    '''
    overbought = df[['overbought']].hvplot.line(
    line_color='red',
    ylabel='Strength',
    width= window_width,
    height= window_height,
    label='Overbought'
    )

    middle = df[['middle']].hvplot.line(
        line_color='purple',
        ylabel='Strength',
        width= window_width,
        height= window_height,
        label='Middle'
    )

    oversold = df[['oversold']].hvplot.line(
        line_color='blue',
        ylabel='Strength',
        width= window_width,
        height= window_height,
        label='Oversold'
    )

    relative_strength = df[['rsi']].hvplot.line(
        title= f'Raw Relative Strength Indicator    Symbol: {company_symbol}',
        ylabel='Strength',
        width= window_width,
        height= window_height,
        label='RSI'
    )

    relative_strength = overbought * middle * oversold * relative_strength

    if output == 'file':
        
        # Save plot charts
        hvplot.save(relative_strength, f'../../../Desktop/new/{file_name}.html')

    elif output == 'screen':

        # Return plot charts
        return relative_strength.opts(xaxis=None, show_legend=True, legend_position=chart_key_position)

def clean_data_from_alpaca(original_df, default_symbols):
    '''Cleans passed data from Alpaca Markets by providing 'timestamp' label
        to index, removes times in index, formats dates in index, removes
        duplicate dates while keep the first instance, and removes date produced on weekends.
    Args:
        original_df (dataframe): Dataframe containing index of dates with column of 'symbol'
        symbols (list): List of symbols found in dataframe
    Returns:
        Dataframe of data with 'timestamp' label for index, times removed from index values, 
        removes data found on weekends, and removes duplicate indices with the initial 
        duplicate value kept in place.
    '''
    # Format timestamps in Index
    listing = []
    for item in original_df.index:
        temp = pd.Timestamp(item).date()
        listing.append(temp)

    # Sets time information in dataframe index to 0 to clean data
    original_df.set_index([listing], inplace=True)
    original_df.index.set_names('timestamp', inplace=True)
    # Remove duplicate timestamp indices
    if len(default_symbols) == 1:
        original_df.reset_index(inplace=True)
        original_df.drop_duplicates('timestamp', keep='first', inplace=True)
        original_df.set_index('timestamp', inplace=True)
    # Remove duplicate timestamp indices for multiple symbols
    else:
        temp_df = pd.DataFrame()
        data_set = pd.DataFrame()
        for ticker in default_symbols:
            data_set = original_df[original_df['symbol'] == ticker]
            data_set.reset_index(inplace=True)
            data_set.drop_duplicates('timestamp', keep='first', inplace=True)
            data_set.set_index('timestamp', inplace=True)
            if temp_df.empty:
                temp_df = data_set
            else:
                temp_df = pd.concat([temp_df, data_set], sort=False)
        original_df = temp_df

    # Removes weekend data
    if len(default_symbols) == 1:
        original_df.reset_index(inplace=True)
        for row in original_df.iterrows():
                if row[1]['timestamp'].weekday() >= 5:
                    original_df.drop(original_df.index[original_df['timestamp'] == row[1]['timestamp']], inplace=True)
        original_df.set_index('timestamp', inplace=True)
    # Remove weekend data for multiple symbols
    else:
        removal_df = pd.DataFrame()
        original_df.reset_index(inplace=True)
        for ticker in default_symbols:
            data_set = original_df[original_df['symbol'] == ticker]
            for row in data_set.iterrows():
                if row[1]['timestamp'].weekday() >= 5:
                    data_set.drop(data_set.index[data_set['timestamp'] == row[1]['timestamp']], inplace=True)
            if removal_df.empty:
                removal_df = data_set
            else:
                removal_df = pd.concat([removal_df, data_set], sort=False)
        original_df = removal_df.set_index('timestamp')

    return original_df

def rsi_breakin_simulating_call_option_trader(def_symbols = ['AMZN'],
    window = 10, research_days = 180, expiration_days = 122, purchase_contracts = 1,
    single_options_contract_price = 9.00, value_of_account = 100000.00):
    '''RSI Simulating Call Option Trader
    Simulates option contract call trading for RSI Break-In Strategy with multiple entry and 
    single exit.  Simulation performs one historical data pull at beginning to limit 
    number of REST calls to Alpaca.
    Args:
        def_symbols (list): List of company ticker symbols to analyze
        window (Int): Historical range to analyze per day
        research_days (Int): Number of days to simulate and trade
        expiration_days (Int): Default number of expiration days for contracts
        purchase_contracts (Int): Number of default contracts to purchase
        single_options_contract_price (Float): Price for a single options contract
        value_of_account (Float): Initial starting value for funding account
    Outputs:
        To screen buy and sell triggers, buy and sell trades executed, and displays
        summary data.
    '''

    # Initialize variables
    default_symbols = def_symbols
    copied_original_df = {}
    data_window = window
    days_to_research = research_days
    total_days_of_data = days_to_research + data_window
    initial_default_days = f'{total_days_of_data} day'
    default_num_expiration_days = expiration_days
    default_num_contracts = purchase_contracts
    default_option_price = single_options_contract_price
    account_value = value_of_account
    account_holdings = []
    option = {}
    max_funds_needed = 0.00
    min_funds_needed = 999999999.99
    account_value_low = 999999999.99
    account_value_high = 0.00
    daily_spending_total = 0.00
    max_spent_in_one_day = 0.00

    # Set date ranges for initial data pull
    initial_data_pull_start_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(initial_default_days)
    
    initial_data_pull_now_date = pd.Timestamp.now(tz='America/New_York').date()
    
    # Tracking initial backtesting start date (excludes previous window of time used for analysis)
    days_to_research_formatted = f'{days_to_research} day'
    backtesting_initial_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

    # Get data from Alpaca Markets
    original_result_df = get_stock_data(default_symbols, '23Hour', initial_data_pull_start_date, initial_data_pull_now_date)

    original_result_df = clean_data_from_alpaca(original_result_df, def_symbols)

    while days_to_research >= 0:

        # Determine new Now date to begin the data window
        days_to_research_formatted = f'{days_to_research} day'
        new_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

        if new_now_date.weekday() < 5:
            # Print current date being analyzed in simulation
            print('Current Now date is: {:%m-%d-%Y}'.format(new_now_date))

            # Simulates brockerage account automatically removing expired option contracts every day
            expiration_position_index = 0
            for position in account_holdings:
                if position['expiration_date'] <= new_now_date:
                    # Pop/remove position from list
                    account_holdings.pop(expiration_position_index)
                expiration_position_index += 1

            # Process data for each symbol
            for item in default_symbols:
                # Make copy of original data to not change base dataframe and reset information for each symbol
                copied_original_df = original_result_df.copy()

                try:
                    # Filter stock data for dates greater than and equal to new start date
                    startdate_filtered_df = copied_original_df.loc[copied_original_df.index >= initial_data_pull_start_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates greater than or equal to initial_data_pull_start_date:\n{e}\nSymbol is:\n{item}\nDataframe is: copied_original_df\n{copied_original_df}')
                    continue
 
                try:
                    # Filter stock data for dates less than and equal to new now date
                    start_and_enddate_filtered_df = startdate_filtered_df.loc[startdate_filtered_df.index <= new_now_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates less than or equal to new_now_date:\n{e}\nSymbol is:\n{item}\nDataframe is: startdate_filtered_df\n{startdate_filtered_df}')
                    continue
 
                try:
                    # Filter stock data for each symbol
                    start_end_symbol_result_df = start_and_enddate_filtered_df[start_and_enddate_filtered_df['symbol'] == item].copy()

                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering spacific symbol data:\n{e}\nSymbol is:\n{item}\nDataframe is: start_and_enddate_filtered_df\n{start_and_enddate_filtered_df}')
                    continue
 
                # Create RSI data for symbol
                signals_df = rsi_breakin(start_end_symbol_result_df, 6, 70, 30)

                try:
                    if (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == -1.0) or (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == 1.0):
                        print('                              Suggestion: Buy  {:<6s} {:8.2f}'.format(item, signals_df['close'][-1]))
                        
                        if default_option_price * (default_num_contracts * 100) <= account_value:
                            option = {
                                'symbol': item,
                                'option_price': default_option_price,
                                'close': signals_df['close'][-1],
                                'num_contracts': default_num_contracts,
                                'purchase_date': signals_df.index[-1],
                                'expiration_date': (signals_df.index[-1] + pd.Timedelta(f'{default_num_expiration_days} day'))
                                }
                            account_holdings.append(option)
                            call_option_purchase_price = default_option_price * (default_num_contracts * 100)
                            account_value -= call_option_purchase_price
                            if account_value < account_value_low:
                                account_value_low = account_value
                            elif account_value > account_value_high:
                                account_value_high = account_value
                            daily_spending_total += call_option_purchase_price
                            print('                                             Bought Position: {:2d} {:<6s}{:8.2f} call option contract(s) for{:8.2f}. Account Value: {:8.2f}'.format(default_num_contracts, item, signals_df['close'][-1], call_option_purchase_price, account_value))
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Buy processes:\n{e}\nSymbol is:\n{item}\nDataframe is: signals_df\n{signals_df}')
                    continue
                try:
                    if (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -2.0) or (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -1.0):
                        print('                              Suggestion: Sell {:<6s}{:8.2f}'.format(item, signals_df['close'][-1]))
                        position_index = 0
                        # Check positions held in brockerage account
                        for position in account_holdings:
                            # Find matching positions to sell and determine execution or allow to expire
                            if position['symbol'] == item:
                                # Position will execute if position plus total cost of initial option contracts purchased is less than current close price
                                # This ensures the initial cost of the purchase is covered in the execution
                                if ((position['close'] * (position['num_contracts'] * 100)) + (position['option_price'] * (position['num_contracts'] * 100))) < (signals_df['close'][-1] * (position['num_contracts'] * 100)):
                                    funds_needed = (position['close'] * (position['num_contracts'] * 100))
                                    if funds_needed < account_value:
                                        profit = ((signals_df['close'][-1] - position['close']) * (position['num_contracts'] * 100))
                                        if funds_needed < min_funds_needed:
                                            min_funds_needed = funds_needed
                                        elif funds_needed > max_funds_needed:
                                            max_funds_needed = funds_needed
                                        account_value += profit
                                        if account_value < account_value_low:
                                            account_value_low = account_value
                                        elif account_value > account_value_high:
                                            account_value_high = account_value
                                        daily_spending_total += funds_needed
                                        print('                                             Exercised Position: {:<6s}{:8.2f} call option and sold all shares at{:8.2f}. Profit:{:8.2f}.  Account Value: {:8.2f}'.format(position['symbol'], position['close'], signals_df['close'][-1], profit, account_value))
                                        # Pop/remove position from list
                                        account_holdings.pop(position_index)
                                    else:
                                        # The account does not have enough funds to exercise the call option position.
                                        print('                                             Held Position: The account value is not large enough to exercise the call option. Add more funds and manually exercise the call option.')
                                else:
                                    # Position price value is higher than current close.
                                    # Current close may rise before expiration of the option contract.
                                    print('                                             Held position: {:<6s}{:8.2f} is greater than current close amount plus the cost of the initial contracts purchased. Account Value:{:8.2f}'.format(position['symbol'], position['close'], account_value))
                            position_index += 1
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Sell processes:\n{e}\nSymbol is:\n{item}\nDataframe is signals_df:\n{signals_df}')
                    continue
        if daily_spending_total != 0.00:
            print('                                                 Daily spending total is: {:8.2f}'.format(daily_spending_total))
        if daily_spending_total > max_spent_in_one_day:
            max_spent_in_one_day = daily_spending_total
        daily_spending_total = 0.00
        days_to_research -= 1

    # Outputs summary information to terminal window
    print('Number of positions still holding: {:d}'.format(len(account_holdings)))
    for position in account_holdings:
        print('Positions held: Symbol: {symbol}, Option Price: {option_price}, Asset Price: {close}, Contracts: {num_contracts}, Purchase: {purchase_date}, Expiration: {expiration_date}'.format(**position))

    print('Account value low: {:8.2f}'.format(account_value_low))
    print('Account value high: {:8.2f}'.format(account_value_high))
    print('Minimum funds needed: {:8.2f}'.format(min_funds_needed))
    print('Maximum funds needed: {:8.2f}'.format(max_funds_needed))
    print('Backtesting analyzed {:%m-%d-%Y} to {:%m-%d-%Y}'.format(backtesting_initial_now_date, new_now_date))
    print('Number of contracts: {:3d}'.format(default_num_contracts))
    print('Account value: {:8.2f}'.format(account_value))
    print('Maximum funds spent in one day: {:8.2f}'.format(max_spent_in_one_day))

def rsi_breakin_simulating_stock_trader(def_symbols = ['AMZN'],
    window = 10, research_days = 180, num_stock_to_purchase = 100,  value_of_account = 100000.00):
    '''RSI Simulating Stock Trader
    Simulates stock trading for RSI Break-In Strategy with multiple entry and 
    single exit.  Simulation performs one historical data pull at beginning to limit 
    number of REST calls to Alpaca.
    Args:
        def_symbols (list): List of company ticker symbols to analyze
        window (Int): Historical range to analyze per day
        research_days (Int): Number of days to simulate and trade
        num_stock_to_purchase (Int): Number of default contracts to purchase
        value_of_account (Float): Initial starting value for funding account
    Outputs:
        To screen buy and sell triggers, buy and sell trades executed, and displays
        summary data.  Function also returns a list containing symbols and account 
        value if liquidated.
    '''

    # Initialize variables
    default_symbols = def_symbols
    copied_original_df = {}
    data_window = window
    days_to_research = research_days
    total_days_of_data = days_to_research + data_window
    initial_default_days = f'{total_days_of_data} day'
    default_num_stocks = num_stock_to_purchase
    account_value = value_of_account
    account_holdings = []
    assets = {}
    max_funds_needed = 0.00
    min_funds_needed = 999999999.99
    account_value_low = 999999999.99
    account_value_high = 0.00
    daily_spending_total = 0.00
    max_spent_in_one_day = 0.00

    # Set date ranges for initial data pull
    initial_data_pull_start_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(initial_default_days)
    
    initial_data_pull_now_date = pd.Timestamp.now(tz='America/New_York').date()
    
    # Tracking initial backtesting start date (excludes previous window of time used for analysis)
    days_to_research_formatted = f'{days_to_research} day'
    backtesting_initial_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

    # Get data from Alpaca Markets
    original_result_df = get_stock_data(default_symbols, '23Hour', initial_data_pull_start_date, initial_data_pull_now_date)

    original_result_df = clean_data_from_alpaca(original_result_df, def_symbols)

    while days_to_research >= 0:

        # Determine new Now date to begin the data window
        days_to_research_formatted = f'{days_to_research} day'
        new_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

        if new_now_date.weekday() < 5:
            # Print current date being analyzed in simulation
            print('Current Now date is: {:%m-%d-%Y}'.format(new_now_date))

            # Process data for each symbol
            for item in default_symbols:
                # Make copy of original data to not change base dataframe and reset information for each symbol
                copied_original_df = original_result_df.copy()

                try:
                    # Filter stock data for dates greater than and equal to new start date
                    startdate_filtered_df = copied_original_df.loc[copied_original_df.index >= initial_data_pull_start_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates greater than or equal to initial_data_pull_start_date:\n{e}\nSymbol is:\n{item}\nDataframe is: copied_original_df\n{copied_original_df}')
                    continue
 
                try:
                    # Filter stock data for dates less than and equal to new now date
                    start_and_enddate_filtered_df = startdate_filtered_df.loc[startdate_filtered_df.index <= new_now_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates less than or equal to new_now_date:\n{e}\nSymbol is:\n{item}\nDataframe is: startdate_filtered_df\n{startdate_filtered_df}')
                    continue
 
                try:
                    # Filter stock data for each symbol
                    start_end_symbol_result_df = start_and_enddate_filtered_df[start_and_enddate_filtered_df['symbol'] == item].copy()

                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering spacific symbol data:\n{e}\nSymbol is:\n{item}\nDataframe is: start_and_enddate_filtered_df\n{start_and_enddate_filtered_df}')
                    continue
 
                # Create RSI data for symbol
                signals_df = rsi_breakin(start_end_symbol_result_df, 6, 70, 30)

                try:
                    if (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == -1.0) or (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == 1.0):
                        print('                              Suggestion: Buy  {:<6s} {:8.2f}'.format(item, signals_df['close'][-1]))
                        funds_needed = signals_df['close'][-1] * default_num_stocks
                        
                        # Check if there are enough funds in account to purchase the triggered asset
                        if funds_needed <= account_value:
                            if funds_needed < min_funds_needed:
                                min_funds_needed = funds_needed
                            elif funds_needed > max_funds_needed:
                                max_funds_needed = funds_needed
                            assets = {
                                'symbol': item,
                                'close': signals_df['close'][-1],
                                'num_stocks': default_num_stocks,
                                'purchase_date': signals_df.index[-1],
                                }
                            account_holdings.append(assets)
                            asset_purchase_price = funds_needed
                            account_value -= asset_purchase_price
                            if account_value < account_value_low:
                                account_value_low = account_value
                            daily_spending_total += asset_purchase_price
                            print('                                             Bought Position: {:2d} {:<6s} stocks for{:8.2f}. Cash Available to Trade: {:8.2f}'.format(default_num_stocks, item, signals_df['close'][-1], account_value))
                        
                        # Not enough funds in account to purchase triggered asset
                        else:
                            print('                                             !!!!!!!!!!!!!!!!!!!!FUNDS NEEDED ARE GREATER THAN ACCOUNT VALUE!!!!!!!!!!Needed: {:10.2f}, Account: {:8.2f}'.format(funds_needed, account_value))
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Buy processes:\n{e}\nSymbol is:\n{item}\nDataframe is: signals_df\n{signals_df}')
                    continue
                try:
                    if (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -2.0) or (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -1.0):
                        print('                              Suggestion: Sell {:<6s}{:8.2f}'.format(item, signals_df['close'][-1]))
                        position_index = 0
                        # Check positions held in brockerage account
                        for position in account_holdings:
                            # Find matching positions to sell and determine execution or allow to expire
                            if position['symbol'] == item:
                                # Position will execute if initial position close price is less than current close price
                                if position['close'] < signals_df['close'][-1]:
                                    sell_value = signals_df['close'][-1] * position['num_stocks']
                                    profit = ((signals_df['close'][-1] - position['close']) * position['num_stocks'])
                                    account_value += sell_value
                                    if account_value > account_value_high:
                                        account_value_high = account_value
                                    print('                                             Sold Position: {:<6s}{:8.2f} stocks.  Sold all shares at{:8.2f}. Profit:{:8.2f}.  Cash Available to Trade: {:8.2f}'.format(position['symbol'], position['close'], signals_df['close'][-1], profit, account_value))
                                    # Pop/remove position from list
                                    account_holdings.pop(position_index)
                                else:
                                    # Position price value is higher than current close.
                                    # Current close may rise at a later date.
                                    print('                                             Held position: {:<6s}{:8.2f} is greater than current close amount {:8.2f}. Cash Available to Trade:{:8.2f}'.format(position['symbol'], position['close'], signals_df['close'][-1], account_value))
                            position_index += 1
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Sell processes:\n{e}\nSymbol is:\n{item}\nDataframe is signals_df:\n{signals_df}')
                    continue
        if daily_spending_total != 0.00:
            print('                                                 Daily spending total is: {:8.2f}'.format(daily_spending_total))
        if daily_spending_total > max_spent_in_one_day:
            max_spent_in_one_day = daily_spending_total
        daily_spending_total = 0.00
        days_to_research -= 1

    # Outputs summary information to terminal window
    print('Number of positions still holding: {:d}'.format(len(account_holdings)))
    liquidate_account_value = account_value
    for position in account_holdings:
        print('Positions held: Symbol: {symbol}, Asset Price: {close}, Number of Stocks: {num_stocks}, Purchased: {purchase_date}'.format(**position), end=' ')
        if len(default_symbols) > 1:
            liquidate_account_value += (original_result_df[original_result_df['symbol'] == position['symbol']]['close'][-1] * num_stock_to_purchase)
            print('Last close price: {:8.2f}'.format(original_result_df[original_result_df['symbol'] == position['symbol']]['close'][-1]))
        else:
            liquidate_account_value += (original_result_df['close'][-1] * num_stock_to_purchase)
            print('Last close price: {:8.2f}'.format(original_result_df['close'][-1]))
    print('Cash value low: {:8.2f}'.format(account_value_low))
    print('Cash value high: {:8.2f}'.format(account_value_high))
    print('Minimum funds needed: {:8.2f}'.format(min_funds_needed))
    print('Maximum funds needed: {:8.2f}'.format(max_funds_needed))
    print('Backtesting analyzed {:%m-%d-%Y} to {:%m-%d-%Y}'.format(backtesting_initial_now_date, new_now_date))
    print('Number of stocks: {:3d}'.format(default_num_stocks))
    print('Cash Available to Trade: {:8.2f}'.format(account_value))
    print('Total account value if liquidated: {:8.2f}'.format(liquidate_account_value))
    print('Maximum funds spent in one day: {:8.2f}'.format(max_spent_in_one_day))

    # Returns liquidated account value and symbols
    return [def_symbols, liquidate_account_value]

def rsi_breakin_simulating_stock_trader_no_pricing_output(def_symbols = ['AMZN'],
    window = 10, research_days = 180):
    '''RSI Simulating Stock Trader
    Simulates stock trading for RSI Break-In Strategy with multiple entry and 
    single exit.  Simulation performs one historical data pull at beginning to limit 
    number of REST calls to Alpaca.
    Args:
        def_symbols (list): List of company ticker symbols to analyze
        window (Int): Historical range to analyze per day
        research_days (Int): Number of days to simulate and trade
        num_stock_to_purchase (Int): Number of default contracts to purchase
        value_of_account (Float): Initial starting value for funding account
    Outputs:
        To screen buy and sell triggers and displays summary data.  Also 
        returns a list containing symbols along with buy sell recommendations  
        for the current day.
    '''

    # Initialize variables
    default_symbols = def_symbols
    copied_original_df = {}
    data_window = window
    days_to_research = research_days
    total_days_of_data = days_to_research + data_window
    initial_default_days = f'{total_days_of_data} day'
    todays_recommendations = []

    # Set date ranges for initial data pull
    initial_data_pull_start_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(initial_default_days)
    
    initial_data_pull_now_date = pd.Timestamp.now(tz='America/New_York').date()
    
    # Tracking initial backtesting start date (excludes previous window of time used for analysis)
    days_to_research_formatted = f'{days_to_research} day'
    backtesting_initial_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

    # Get data from Alpaca Markets
    original_result_df = get_stock_data(default_symbols, '23Hour', initial_data_pull_start_date, initial_data_pull_now_date)

    original_result_df = clean_data_from_alpaca(original_result_df, def_symbols)

    while days_to_research >= 0:
        # Determine new Now date to begin the data window
        days_to_research_formatted = f'{days_to_research} day'
        new_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

        if new_now_date.weekday() < 5:
            # Print current date being analyzed in simulation
            print('Current Now date is: {:%m-%d-%Y}'.format(new_now_date))

            # Process data for each symbol
            for item in default_symbols:
                # Make copy of original data to not change base dataframe and reset information for each symbol
                copied_original_df = original_result_df.copy()

                try:
                    # Filter stock data for dates greater than and equal to new start date
                    startdate_filtered_df = copied_original_df.loc[copied_original_df.index >= initial_data_pull_start_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates greater than or equal to initial_data_pull_start_date:\n{e}\nSymbol is:\n{item}\nDataframe is: copied_original_df\n{copied_original_df}')
                    continue
 
                try:
                    # Filter stock data for dates less than and equal to new now date
                    start_and_enddate_filtered_df = startdate_filtered_df.loc[startdate_filtered_df.index <= new_now_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates less than or equal to new_now_date:\n{e}\nSymbol is:\n{item}\nDataframe is: startdate_filtered_df\n{startdate_filtered_df}')
                    continue
 
                try:
                    # Filter stock data for each symbol
                    start_end_symbol_result_df = start_and_enddate_filtered_df[start_and_enddate_filtered_df['symbol'] == item].copy()

                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering spacific symbol data:\n{e}\nSymbol is:\n{item}\nDataframe is: start_and_enddate_filtered_df\n{start_and_enddate_filtered_df}')
                    continue
 
                # Create RSI data for symbol
                signals_df = rsi_breakin(start_end_symbol_result_df, 6, 70, 30)

                # Buy triggers
                try:
                    if (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == -1.0) or (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == 1.0):
                        print('                              Suggestion: Buy  {:<6s} {:8.2f}'.format(item, signals_df['close'][-1]))
                        
                        # Capture current day's recommended buys
                        if days_to_research == 1:
                            todays_recommendations.append('Buy  {:<6s} {:8.2f}'.format(item, signals_df['close'][-1]))
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Buy processes:\n{e}\nSymbol is:\n{item}\nDataframe is: signals_df\n{signals_df}')
                    continue

                # Sell Triggers
                try:
                    if (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -2.0) or (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -1.0):
                        print('                              Suggestion: Sell {:<6s}{:8.2f}'.format(item, signals_df['close'][-1]))

                        # Capture current day's recommended sells
                        if days_to_research == 1:
                            todays_recommendations.append('Sell {:<6s}{:8.2f}'.format(item, signals_df['close'][-1]))
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Sell processes:\n{e}\nSymbol is:\n{item}\nDataframe is signals_df:\n{signals_df}')
                    continue
        days_to_research -= 1

    # Outputs summary information to terminal window
    print('Backtesting analyzed {:%m-%d-%Y} to {:%m-%d-%Y}'.format(backtesting_initial_now_date, new_now_date))

    # Print today's recommendations if any
    if len(todays_recommendations) > 0:
        for recommendation in todays_recommendations:
            print(recommendation)
    else:
        print('No recommendations today')
        todays_recommendations.append('No recommendations today')

    # Returns list of symbols scanned and today's recommendations
    return [def_symbols, todays_recommendations]

def rsi_breakout_simulating_call_option_trader(def_symbols = ['AMZN'],
    window = 10, research_days = 180, expiration_days = 122, purchase_contracts = 1,
    single_options_contract_price = 9.00, value_of_account = 100000.00):
    '''RSI Simulating Call Option Trader
    Simulates option contract call trading for RSI Break-Out Strategy with multiple entry and 
    single exit.  Simulation performs one historical data pull at beginning to limit 
    number of REST calls to Alpaca.
    Args:
        def_symbols (list): List of company ticker symbols to analyze
        window (Int): Historical range to analyze per day
        research_days (Int): Number of days to simulate and trade
        expiration_days (Int): Default number of expiration days for contracts
        purchase_contracts (Int): Number of default contracts to purchase
        single_options_contract_price (Float): Price for a single options contract
        value_of_account (Float): Initial starting value for funding account
    Outputs:
        To screen buy and sell triggers, buy and sell trades executed, and displays
        summary data.
    '''

    # Initialize variables
    default_symbols = def_symbols
    copied_original_df = {}
    data_window = window
    days_to_research = research_days
    total_days_of_data = days_to_research + data_window
    initial_default_days = f'{total_days_of_data} day'
    default_num_expiration_days = expiration_days
    default_num_contracts = purchase_contracts
    default_option_price = single_options_contract_price
    account_value = value_of_account
    account_holdings = []
    option = {}
    max_funds_needed = 0.00
    min_funds_needed = 999999999.99
    account_value_low = 999999999.99
    account_value_high = 0.00
    daily_spending_total = 0.00
    max_spent_in_one_day = 0.00

    # Set date ranges for initial data pull
    initial_data_pull_start_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(initial_default_days)
    
    initial_data_pull_now_date = pd.Timestamp.now(tz='America/New_York').date()
    
    # Tracking initial backtesting start date (excludes previous window of time used for analysis)
    days_to_research_formatted = f'{days_to_research} day'
    backtesting_initial_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

    # Get data from Alpaca Markets
    original_result_df = get_stock_data(default_symbols, '23Hour', initial_data_pull_start_date, initial_data_pull_now_date)

    original_result_df = clean_data_from_alpaca(original_result_df, def_symbols)

    while days_to_research >= 0:

        # Determine new Now date to begin the data window
        days_to_research_formatted = f'{days_to_research} day'
        new_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

        if new_now_date.weekday() < 5:
            # Print current date being analyzed in simulation
            print('Current Now date is: {:%m-%d-%Y}'.format(new_now_date))

            # Simulates brockerage account automatically removing expired option contracts every day
            expiration_position_index = 0
            for position in account_holdings:
                if position['expiration_date'] <= new_now_date:
                    # Pop/remove position from list
                    account_holdings.pop(expiration_position_index)
                expiration_position_index += 1

            # Process data for each symbol
            for item in default_symbols:
                # Make copy of original data to not change base dataframe and reset information for each symbol
                copied_original_df = original_result_df.copy()

                try:
                    # Filter stock data for dates greater than and equal to new start date
                    startdate_filtered_df = copied_original_df.loc[copied_original_df.index >= initial_data_pull_start_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates greater than or equal to initial_data_pull_start_date:\n{e}\nSymbol is:\n{item}\nDataframe is: copied_original_df\n{copied_original_df}')
                    continue
 
                try:
                    # Filter stock data for dates less than and equal to new now date
                    start_and_enddate_filtered_df = startdate_filtered_df.loc[startdate_filtered_df.index <= new_now_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates less than or equal to new_now_date:\n{e}\nSymbol is:\n{item}\nDataframe is: startdate_filtered_df\n{startdate_filtered_df}')
                    continue
 
                try:
                    # Filter stock data for each symbol
                    start_end_symbol_result_df = start_and_enddate_filtered_df[start_and_enddate_filtered_df['symbol'] == item].copy()

                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering spacific symbol data:\n{e}\nSymbol is:\n{item}\nDataframe is: start_and_enddate_filtered_df\n{start_and_enddate_filtered_df}')
                    continue
 
                # Create RSI data for symbol
                signals_df = rsi_breakout(start_end_symbol_result_df, 6, 70, 30)

                try:
                    if (signals_df['signal'][-2] == 1.0 and signals_df['entry/exit'][-2] == 1.0) or (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -1.0):
                        print('                              Suggestion: Buy  {:<6s} {:8.2f}'.format(item, signals_df['close'][-1]))
                        
                        if default_option_price * (default_num_contracts * 100) <= account_value:
                            option = {
                                'symbol': item,
                                'option_price': default_option_price,
                                'close': signals_df['close'][-1],
                                'num_contracts': default_num_contracts,
                                'purchase_date': signals_df.index[-1],
                                'expiration_date': (signals_df.index[-1] + pd.Timedelta(f'{default_num_expiration_days} day'))
                                }
                            account_holdings.append(option)
                            call_option_purchase_price = default_option_price * (default_num_contracts * 100)
                            account_value -= call_option_purchase_price
                            if account_value < account_value_low:
                                account_value_low = account_value
                            elif account_value > account_value_high:
                                account_value_high = account_value
                            daily_spending_total += call_option_purchase_price
                            print('                                             Bought Position: {:2d} {:<6s}{:8.2f} call option contract(s) for{:8.2f}. Account Value: {:8.2f}'.format(default_num_contracts, item, signals_df['close'][-1], call_option_purchase_price, account_value))
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Buy processes:\n{e}\nSymbol is:\n{item}\nDataframe is: signals_df\n{signals_df}')
                    continue
                try:
                    if (signals_df['signal'][-2] == 1.0 and signals_df['entry/exit'][-2] == 2.0) or (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == 1.0):
                        print('                              Suggestion: Sell {:<6s}{:8.2f}'.format(item, signals_df['close'][-1]))
                        position_index = 0
                        # Check positions held in brockerage account
                        for position in account_holdings:
                            # Find matching positions to sell and determine execution or allow to expire
                            if position['symbol'] == item:
                                # Position will execute if position plus total cost of initial option contracts purchased is less than current close price
                                # This ensures the initial cost of the purchase is covered in the execution
                                if ((position['close'] * (position['num_contracts'] * 100)) + (position['option_price'] * (position['num_contracts'] * 100))) < (signals_df['close'][-1] * (position['num_contracts'] * 100)):
                                    funds_needed = (position['close'] * (position['num_contracts'] * 100))
                                    if funds_needed < account_value:
                                        profit = ((signals_df['close'][-1] - position['close']) * (position['num_contracts'] * 100))
                                        if funds_needed < min_funds_needed:
                                            min_funds_needed = funds_needed
                                        elif funds_needed > max_funds_needed:
                                            max_funds_needed = funds_needed
                                        account_value += profit
                                        if account_value < account_value_low:
                                            account_value_low = account_value
                                        elif account_value > account_value_high:
                                            account_value_high = account_value
                                        daily_spending_total += funds_needed
                                        print('                                             Exercised Position: {:<6s}{:8.2f} call option and sold all shares at{:8.2f}. Profit:{:8.2f}.  Account Value: {:8.2f}'.format(position['symbol'], position['close'], signals_df['close'][-1], profit, account_value))
                                        # Pop/remove position from list
                                        account_holdings.pop(position_index)
                                    else:
                                        # The account does not have enough funds to exercise the call option position.
                                        print('                                             Held Position: The account value is not large enough to exercise the call option. Add more funds and manually exercise the call option.')
                                else:
                                    # Position price value is higher than current close.
                                    # Current close may rise before expiration of the option contract.
                                    print('                                             Held position: {:<6s}{:8.2f} is greater than current close amount plus the cost of the initial contracts purchased. Account Value:{:8.2f}'.format(position['symbol'], position['close'], account_value))
                            position_index += 1
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Sell processes:\n{e}\nSymbol is:\n{item}\nDataframe is signals_df:\n{signals_df}')
                    continue
        if daily_spending_total != 0.00:
            print('                                                 Daily spending total is: {:8.2f}'.format(daily_spending_total))
        if daily_spending_total > max_spent_in_one_day:
            max_spent_in_one_day = daily_spending_total
        daily_spending_total = 0.00
        days_to_research -= 1

    # Outputs summary information to terminal window
    print('Number of positions still holding: {:d}'.format(len(account_holdings)))
    for position in account_holdings:
        print('Positions held: Symbol: {symbol}, Option Price: {option_price}, Asset Price: {close}, Contracts: {num_contracts}, Purchase: {purchase_date}, Expiration: {expiration_date}'.format(**position))

    print('Account value low: {:8.2f}'.format(account_value_low))
    print('Account value high: {:8.2f}'.format(account_value_high))
    print('Minimum funds needed: {:8.2f}'.format(min_funds_needed))
    print('Maximum funds needed: {:8.2f}'.format(max_funds_needed))
    print('Backtesting analyzed {:%m-%d-%Y} to {:%m-%d-%Y}'.format(backtesting_initial_now_date, new_now_date))
    print('Number of contracts: {:3d}'.format(default_num_contracts))
    print('Account value: {:8.2f}'.format(account_value))
    print('Maximum funds spent in one day: {:8.2f}'.format(max_spent_in_one_day))

def rsi_breakout_simulating_stock_trader(def_symbols = ['AMZN'],
    window = 10, research_days = 180, num_stock_to_purchase = 100,  value_of_account = 100000.00):
    '''RSI Simulating Stock Trader
    Simulates stock trading for RSI Break-Out Strategy with multiple entry and 
    single exit.  Simulation performs one historical data pull at beginning to limit 
    number of REST calls to Alpaca.
    Args:
        def_symbols (list): List of company ticker symbols to analyze
        window (Int): Historical range to analyze per day
        research_days (Int): Number of days to simulate and trade
        num_stock_to_purchase (Int): Number of default contracts to purchase
        value_of_account (Float): Initial starting value for funding account
    Outputs:
        To screen buy and sell triggers, buy and sell trades executed, and displays
        summary data.
    '''

    # Initialize variables
    default_symbols = def_symbols
    copied_original_df = {}
    data_window = window
    days_to_research = research_days
    total_days_of_data = days_to_research + data_window
    initial_default_days = f'{total_days_of_data} day'
    default_num_stocks = num_stock_to_purchase
    account_value = value_of_account
    account_holdings = []
    assets = {}
    max_funds_needed = 0.00
    min_funds_needed = 999999999.99
    account_value_low = 999999999.99
    account_value_high = 0.00
    daily_spending_total = 0.00
    max_spent_in_one_day = 0.00

    # Set date ranges for initial data pull
    initial_data_pull_start_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(initial_default_days)
    
    initial_data_pull_now_date = pd.Timestamp.now(tz='America/New_York').date()
    
    # Tracking initial backtesting start date (excludes previous window of time used for analysis)
    days_to_research_formatted = f'{days_to_research} day'
    backtesting_initial_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

    # Get data from Alpaca Markets
    original_result_df = get_stock_data(default_symbols, '23Hour', initial_data_pull_start_date, initial_data_pull_now_date)

    original_result_df = clean_data_from_alpaca(original_result_df, def_symbols)

    while days_to_research >= 0:

        # Determine new Now date to begin the data window
        days_to_research_formatted = f'{days_to_research} day'
        new_now_date = pd.Timestamp.now(tz='America/New_York').date() - pd.Timedelta(days_to_research_formatted)

        if new_now_date.weekday() < 5:
            # Print current date being analyzed in simulation
            print('Current Now date is: {:%m-%d-%Y}'.format(new_now_date))

            # Process data for each symbol
            for item in default_symbols:
                # Make copy of original data to not change base dataframe and reset information for each symbol
                copied_original_df = original_result_df.copy()

                try:
                    # Filter stock data for dates greater than and equal to new start date
                    startdate_filtered_df = copied_original_df.loc[copied_original_df.index >= initial_data_pull_start_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates greater than or equal to initial_data_pull_start_date:\n{e}\nSymbol is:\n{item}\nDataframe is: copied_original_df\n{copied_original_df}')
                    continue
 
                try:
                    # Filter stock data for dates less than and equal to new now date
                    start_and_enddate_filtered_df = startdate_filtered_df.loc[startdate_filtered_df.index <= new_now_date].copy()
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering dates less than or equal to new_now_date:\n{e}\nSymbol is:\n{item}\nDataframe is: startdate_filtered_df\n{startdate_filtered_df}')
                    continue
 
                try:
                    # Filter stock data for each symbol
                    start_end_symbol_result_df = start_and_enddate_filtered_df[start_and_enddate_filtered_df['symbol'] == item].copy()

                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to filtering spacific symbol data:\n{e}\nSymbol is:\n{item}\nDataframe is: start_and_enddate_filtered_df\n{start_and_enddate_filtered_df}')
                    continue
 
                # Create RSI data for symbol
                signals_df = rsi_breakout(start_end_symbol_result_df, 6, 70, 30)

                try:
                    if (signals_df['signal'][-2] == 1.0 and signals_df['entry/exit'][-2] == 1.0) or (signals_df['signal'][-2] == -1.0 and signals_df['entry/exit'][-2] == -1.0):
                        print('                              Suggestion: Buy  {:<6s} {:8.2f}'.format(item, signals_df['close'][-1]))
                        funds_needed = signals_df['close'][-1] * default_num_stocks
                        if funds_needed <= account_value:
                            if funds_needed < min_funds_needed:
                                min_funds_needed = funds_needed
                            elif funds_needed > max_funds_needed:
                                max_funds_needed = funds_needed
                            assets = {
                                'symbol': item,
                                'close': signals_df['close'][-1],
                                'num_stocks': default_num_stocks,
                                'purchase_date': signals_df.index[-1],
                                }
                            account_holdings.append(assets)
                            asset_purchase_price = funds_needed
                            account_value -= asset_purchase_price
                            if account_value < account_value_low:
                                account_value_low = account_value
                            daily_spending_total += asset_purchase_price
                            print('                                             Bought Position: {:2d} {:<6s} stocks for{:8.2f}. Cash Available to Trade: {:8.2f}'.format(default_num_stocks, item, signals_df['close'][-1], account_value))
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Buy processes:\n{e}\nSymbol is:\n{item}\nDataframe is: signals_df\n{signals_df}')
                    continue
                try:
                    if (signals_df['signal'][-2] == 1.0 and signals_df['entry/exit'][-2] == 2.0) or (signals_df['signal'][-2] == 0.0 and signals_df['entry/exit'][-2] == 1.0):
                        print('                              Suggestion: Sell {:<6s}{:8.2f}'.format(item, signals_df['close'][-1]))
                        position_index = 0
                        # Check positions held in brockerage account
                        for position in account_holdings:
                            # Find matching positions to sell and determine execution or allow to expire
                            if position['symbol'] == item:
                                # Position will execute if initial position close price is less than current close price
                                if position['close'] < signals_df['close'][-1]:
                                        sell_value = signals_df['close'][-1] * position['num_stocks']
                                        profit = ((signals_df['close'][-1] - position['close']) * position['num_stocks'])
                                        account_value += sell_value
                                        if account_value > account_value_high:
                                            account_value_high = account_value
                                        print('                                             Sold Position: {:<6s}{:8.2f} stocks.  Sold all shares at{:8.2f}. Profit:{:8.2f}.  Cash Available to Trade: {:8.2f}'.format(position['symbol'], position['close'], signals_df['close'][-1], profit, account_value))
                                        # Pop/remove position from list
                                        account_holdings.pop(position_index)
                                else:
                                    # Position price value is higher than current close.
                                    # Current close may rise at a later date.
                                    print('                                             Held position: {:<6s}{:8.2f} is greater than current close amount {:8.2f}. Cash Available to Trade:{:8.2f}'.format(position['symbol'], position['close'], signals_df['close'][-1], account_value))
                            position_index += 1
                except Exception as e:
                    # Log exceptions and continue to next symbol
                    logger.info(f'\nIssue that was thrown is related to Sell processes:\n{e}\nSymbol is:\n{item}\nDataframe is signals_df:\n{signals_df}')
                    continue
        if daily_spending_total != 0.00:
            print('                                                 Daily spending total is: {:8.2f}'.format(daily_spending_total))
        if daily_spending_total > max_spent_in_one_day:
            max_spent_in_one_day = daily_spending_total
        daily_spending_total = 0.00
        days_to_research -= 1

    # Outputs summary information to terminal window
    print('Number of positions still holding: {:d}'.format(len(account_holdings)))
    liquidate_account_value = account_value
    for position in account_holdings:
        print('Positions held: Symbol: {symbol}, Asset Price: {close}, Number of Stocks: {num_stocks}, Purchased: {purchase_date}'.format(**position), end=' ')
        if len(default_symbols) > 1:
            liquidate_account_value += (original_result_df[original_result_df['symbol'] == position['symbol']]['close'][-1] * num_stock_to_purchase)
            print('Last close price: {:8.2f}'.format(original_result_df[original_result_df['symbol'] == position['symbol']]['close'][-1]))
        else:
            liquidate_account_value += (original_result_df['close'][-1] * num_stock_to_purchase)
            print('Last close price: {:8.2f}'.format(original_result_df['close'][-1]))
    print('Cash value low: {:8.2f}'.format(account_value_low))
    print('Cash value high: {:8.2f}'.format(account_value_high))
    print('Minimum funds needed: {:8.2f}'.format(min_funds_needed))
    print('Maximum funds needed: {:8.2f}'.format(max_funds_needed))
    print('Backtesting analyzed {:%m-%d-%Y} to {:%m-%d-%Y}'.format(backtesting_initial_now_date, new_now_date))
    print('Number of stocks: {:3d}'.format(default_num_stocks))
    print('Cash Available to Trade: {:8.2f}'.format(account_value))
    print('Total account value if liquidated: {:8.2f}'.format(liquidate_account_value))
    print('Maximum funds spent in one day: {:8.2f}'.format(max_spent_in_one_day))
