import sys
sys.path.append("/Users/Jimo/Python Scripts/Crypto_Currency_Trading")

import re
import numpy as np
import pandas as pd

from Utils.Data import Feeder
from Utils.Metrics import Metrics

#########################
# Utility Functions
#########################

def calc_pattern_freq(series=[], pattern_len=6):
    if series is None or series == []:
        print("Input series is not provided or empty.")
        return

    series_freq_dict = {}
    series_len = len(series)

    for i in range(0, series_len - pattern_len + 1):
        pattern = tuple(series[i: i + pattern_len])

        if pattern in series_freq_dict:
            series_freq_dict[pattern] += 1
        else:
            series_freq_dict[pattern] = 1

    for pattern in series_freq_dict:
        series_freq_dict[pattern] /= (series_len - pattern_len + 1)

    return series_freq_dict


def calc_trade_signal(pattern_freq_dict=None, window_price_pattern=None, curr_position=None):
    if any([arg is None for arg in [pattern_freq_dict, window_price_pattern, curr_position]]):
        print("Pattern frequency dictionary, window price pattern, or current position is not given.")
        return

    # BIT coin shorting is not allowed in all exchanges. We disallow shorting in our strategy.
    if curr_position < 0:
        print("Short position is disallowed. Current position " + str(curr_position) + " is not valid.")
        return

    # Take pattern from n-1 point rolling window, and create n point up or down pattern.
    up_pattern = window_price_pattern + (1,)
    down_pattern = window_price_pattern + (-1,)

    # If pattern was never discovered in history, its frequency is set to 0, for comparison purposes.
    up_freq = pattern_freq_dict[up_pattern] if up_pattern in pattern_freq_dict else 0
    down_freq = pattern_freq_dict[down_pattern] if down_pattern in pattern_freq_dict else 0

    # Previous validation ensures curr_position is non-negative.
    if curr_position == 0:
        # We hold no position
        if up_freq > down_freq:
            # Predicting higher price -> buy
            return "BUY"
        elif up_freq < down_freq:
            # Predicting lower or equal price -> no action
            return "NO ACTION"
        else:
            # No decision
            return "NO ACTION"
    elif curr_position > 0:
        # We hold long positions
        if up_freq > down_freq:
            # Predicting higher price -> no action
            return "NO ACTION"
        elif up_freq < down_freq:
            # Predicting lower or equal price -> sell all position
            return "SELL"
        else:
            # No decision
            return "NO ACTION"


def calc_trade_signal_allow_short(pattern_freq_dict=None, window_price_pattern=None, curr_position=None):
    if any([arg is None for arg in [pattern_freq_dict, window_price_pattern, curr_position]]):
        print("Pattern frequency dictionary, window price pattern, or current position is not given.")
        return

    # Allow bitcoin shorting.

    # Take pattern from n-1 point rolling window, and create n point up or down pattern.
    up_pattern = window_price_pattern + (1,)
    down_pattern = window_price_pattern + (-1,)

    # If pattern was never discovered in history, its frequency is set to 0, for comparison purposes.
    up_freq = pattern_freq_dict[up_pattern] if up_pattern in pattern_freq_dict else 0
    down_freq = pattern_freq_dict[down_pattern] if down_pattern in pattern_freq_dict else 0

    # Shorting is allowed, and thus position could be negative.
    if curr_position == 0:
        # We hold no position
        if up_freq > down_freq:
            # Predicting higher price
            return "BUY"
        elif up_freq < down_freq:
            # Predicting lower price
            return "SHORT"
        else:
            # No valid prediction
            return "NO ACTION"
    elif curr_position > 0:
        # We hold long positions
        if up_freq > down_freq:
            # Predicting higher price
            return "NO ACTION"
        elif up_freq < down_freq:
            # Predicting lower price
            return "CLOSE, SHORT"
        else:
            # No valid prediction
            return "NO ACTION"
    else:
        # We hold short positions
        if up_freq > down_freq:
            # Predicting higher price
            return "CLOSE, BUY"
        elif up_freq < down_freq:
            # Predicting lower price
            return "NO ACTION"
        else:
            # No valid prediction
            return "NO ACTION"


def execute_slm(pattern_freq_dict=None, price_data_df=None, window_len=5, initial_amount=0):
    """
            Backtest the trading strategy

            Parameters
            ----------
            pattern_freq_dict : dict,
                dictionary containing historical patterns and their frequency

            price_data_df: dataframe,
                dataframe containing historical price data up to now,
                it has columns: time_exchange, bid1_price, ask1_price, mid_price, mid_price_increments, mid_price_direction

            initial_amount:
                initial number of funding

            Returns
            -------
    """

    if any([arg is None for arg in (pattern_freq_dict, price_data_df, initial_amount)]):
        print("Pattern frequency dictionary, price data, or initial amount is not provided.")
        return

    price_data_len = len(price_data_df)

    # No position: 0
    # Hold position: positive
    # Short position: negative (disabled in our strategy)
    curr_position = 0
    curr_amount = initial_amount

    buy_count = 0
    sell_count = 0

    for i in range(0, price_data_len - window_len):
        # print("Processing the " + str(i) + "th of the " + str(price_data_len - window_len) + " records.")

        # Historical price data from rolling window
        window_price_data_df = price_data_df.iloc[i:i + window_len, :]
        window_pattern = tuple(window_price_data_df['mid_price_direction'].tolist())

        # Generate trading signal
        trade_signal = calc_trade_signal(pattern_freq_dict, window_pattern, curr_position)

        # Execute trade
        if trade_signal == "BUY":
            # Buy at ask 1 at current moment
            buy_price = window_price_data_df["ask1_price"].tolist()[-1]
            buy_size = 1

            # Calculate position and amount
            curr_position += buy_size
            curr_amount -= buy_price * buy_size

            buy_count += 1
        elif trade_signal == "SELL":
            # Sell at bid 1 at current moment
            sell_price = window_price_data_df["bid1_price"].tolist()[-1]
            sell_size = 1

            # Calculate position and amount
            curr_position -= sell_size
            curr_amount += sell_price * sell_size

            sell_count += 1
        elif trade_signal == "NO ACTION":
            # Maintain current position
            curr_position = curr_position
            curr_amount = curr_amount

    profit = curr_amount - initial_amount

    return (profit, buy_count, sell_count)


def execute_slm_allow_short(pattern_freq_dict=None, price_data_df=None, window_len=5, initial_amount=0):
    """
        Backtest the trading strategy

        Parameters
        ----------
        pattern_freq_dict : dict,
            dictionary containing historical patterns and their frequency

        price_data_df: dataframe,
            dataframe containing historical price data up to now,
            it has columns: time_exchange, bid1_price, ask1_price, mid_price, mid_price_increments, mid_price_direction

        initial_amount:
            initial number of funding

        Returns
        -------
    """
    if any([arg is None for arg in (pattern_freq_dict, price_data_df, initial_amount)]):
        print("Pattern frequency dictionary, price data, or initial amount is not provided.")
        return

    price_data_len = len(price_data_df)

    # No position: 0
    # Hold position: positive
    # Short position: negative
    curr_position = 0
    curr_amount = initial_amount

    buy_count = 0
    short_count = 0

    for i in range(0, price_data_len - window_len):
        # print("Processing the " + str(i) + "th of the " + str(price_data_len - window_len) + " records.")

        # Historical price data from rolling window
        window_price_data_df = price_data_df.iloc[i:i + window_len, :]
        window_pattern = tuple(window_price_data_df['mid_price_direction'].tolist())

        # Generate trading signal
        trade_signal = calc_trade_signal_allow_short(pattern_freq_dict, window_pattern, curr_position)

        # Execute trade
        if trade_signal == "BUY":
            # Buy at ask 1 at current moment
            buy_price = window_price_data_df["ask1_price"].tolist()[-1]
            buy_size = 1

            # Calculate position and amount
            curr_position += buy_size
            curr_amount -= buy_price * buy_size

            buy_count += 1
        elif trade_signal == "SHORT":
            # Short at bid 1 at current moment
            short_price = window_price_data_df["bid1_price"].tolist()[-1]
            short_size = 1

            # Calculate position and amount
            curr_position -= short_size
            curr_amount += short_price * short_size

            short_count += 1
        elif trade_signal == "NO ACTION":
            # Maintain current position
            curr_position = curr_position
            curr_amount = curr_amount
        elif trade_signal == "CLOSE, BUY":
            # Buy at ask 1 at current moment
            buy_price = window_price_data_df["ask1_price"].tolist()[-1]
            # This happens when curr_position is negative
            # Close current (short) position; then enter long position
            assert curr_position < 0, "CLOSE, BUY should only happen under negative position."
            buy_size = abs(curr_position) + 1

            # Calculate position and amount
            curr_position += buy_size
            curr_amount -= buy_price * buy_size

            buy_count += 1
        elif trade_signal == "CLOSE, SHORT":
            # Short at bid 1 at current moment
            short_price = window_price_data_df["bid1_price"].tolist()[-1]
            # This happens when curr_position is positive
            # Close current(long) position; then enter short position
            assert curr_position > 0, "CLOSE, SHORT should only happen under positive position."
            short_size = abs(curr_position) + 1

            # Calculate position and amount
            curr_position -= short_size
            curr_amount += short_price * short_size

            short_count += 1

    profit = curr_amount - initial_amount

    return (profit, buy_count, short_count)


def create_unique_price_data_df(depth_data=None, quote_data=None, trade_data=None):
    if any([depth_data is None, quote_data is None, trade_data is None]):
        print("Depth data, quote data, or trade data is not provided.")
        return

    analysis = Metrics(depth_data, quote_data, trade_data)
    mid_prices = analysis.depth_data["mid_price"]
    mid_price_df = mid_prices.to_frame().reset_index()

    bid_ask_df = Depth_data.filter(regex="(ask|bid)[1]_price").reset_index()

    price_df = pd.merge(bid_ask_df, mid_price_df, how='inner', on='time_exchange')

    price_increments = mid_prices - mid_prices.shift(1)
    price_increments_df = price_increments.dropna().to_frame().reset_index()
    price_increments_df = price_increments_df.rename(columns={'mid_price': 'mid_price_increments'})
    price_increments_df["mid_price_direction"] = -1
    price_increments_df["mid_price_direction"].loc[price_increments_df["mid_price_increments"] > 0] = 1

    price_data_df = pd.merge(price_df, price_increments_df, how='inner', on='time_exchange')

    # Remove rows with duplicated mid-price
    # If we don't remove duplicates, the equal price will dominate historical patterns.
    # All prediction will lead to "price down or equal", causing "no action".
    # Therefore, we remove duplicated mid-prices and only consider effective price moves.
    unique_price_data_df = price_data_df.drop_duplicates(subset=["mid_price"], keep='first')

    return unique_price_data_df


def tune_execute_slm(price_data_df=None, train_percentage = 0.8, pattern_length = 6, allow_short=False):
    # Another parameter to tune, which is determined by pattern length.
    window_length = pattern_length - 1

    if price_data_df is None:
        print("Price data is not provided.")
        return

    # Train, test split.
    train_size = int(len(price_data_df) * train_percentage)
    train_price_data_df = price_data_df[0: train_size]
    test_price_data_df = price_data_df[train_size:]

    # Generate historical pattern dictionary via train data.
    pattern_freq_dict = calc_pattern_freq(train_price_data_df["mid_price_direction"].tolist(), pattern_length)

    # Execute SLM strategy via test data.
    if allow_short:
        profit, buy_count, sell_count = execute_slm_allow_short(
            pattern_freq_dict,
            test_price_data_df,
            window_len=window_length,
            initial_amount=1000000
        )
    else:
        profit, buy_count, sell_count = execute_slm(
            pattern_freq_dict,
            test_price_data_df,
            window_len=window_length,
            initial_amount=1000000)

    print("Profit is: " + str(profit) + "...")
    print("Buy order counts: " + str(buy_count) + "...")
    print("Sell order counts: " + str(sell_count) + "...")

    return (profit, buy_count, sell_count)


#########################
# Unit Test
#########################

def test_calc_series_freq():
    series = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    series_freq_dict = calc_pattern_freq(series, 3)

    for pattern, count in series_freq_dict.items():
        print("Pattern: " + str(pattern))
        print("Count: " + str(count))

    print("Test for calc_series_freq is done.")


def test_calc_trade_signal():
    pattern_freq_dict = {
        (-1, -1, 1): 1,
        (-1, 1, -1): 3,
        (-1, 1, 1): 2,
        (1, -1, -1): 3,
        (1, -1, 1): 3,
        (1, 1, -1): 1,
        (1, 1, 1): 3
    }

    curr_positions = [-1, 0, 1]

    win_patterns = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for curr_position in curr_positions:
        for win_pattern in win_patterns:
            res = calc_trade_signal(pattern_freq_dict, win_pattern, curr_position)
            print(res)

    print("Test for calc_trade_signal is done.")


#########################
# Main Function
#########################

if __name__ == "__main__":
    # test_calc_series_freq()

    # test_calc_trade_signal()

    Depth_data = Feeder("../Data/BINANCE_SPOT_BTC_USDT_01012018_depth.csv").get()
    Quote_data = Feeder("../Data/BINANCE_SPOT_BTC_USDT_01012018_quote.csv").get()
    Trade_data = Feeder("../Data/BINANCE_SPOT_BTC_USDT_01012018_trade.csv").get()

    unique_price_data_df = create_unique_price_data_df(Depth_data, Quote_data, Trade_data)

    train_percent = 0.8
    pattern_len = 6
    tune_execute_slm(
        price_data_df=unique_price_data_df,
        train_percentage=train_percent,
        pattern_length=pattern_len,
        allow_short=True)

    print("Test is done.")

