import pandas as pd
import numpy as np
from re import findall


class Analysis:
    def __init__(self, depth_data, quote_data, trade_data):
        """
        Parameters
        ----------
        depth_data : pd.DataFrame
        quote_data : pd.DataFrame
        trade_data : pd.DataFrame
        """
        self.depth_data = depth_data
        self.quote_data = quote_data
        self.trade_data = trade_data

        # Generate the mid price and the last price
        self.depth_data["mid_price"] = (self.depth_data["ask1_price"] + \
                                        self.depth_data["bid1_price"]) / 2
        last_trade_data = self.trade_data[~self.trade_data.index.duplicated(keep="last")].reindex(self.depth_data.index,
                                                                                                  method="ffill")
        self.depth_data["last_price"] = last_trade_data["price"]
        self.depth_data["last_size"] = last_trade_data["size"]

    @property
    def dims(self):
        """
        Get the dimension of order books
        Returns
        -------
        int, int : num of observations, num of price levels
        """
        num_observations = self.depth_data.shape[0]
        num_levels = sum([1 if findall(r"ask\d+_price", x) else 0 for x in self.depth_data.columns])
        return num_observations, num_levels

    def get_price(self, side):
        """
        Obtain price levels of a particular trading side with renamed column names
        Parameters
        ----------
        side : {"bid", "ask"}
            Trading sides
        Returns
        -------
            pd.DataFrame
        """
        assert side in {"bid", "ask"}, "Trading sides should be {'bid', 'ask'}"
        return self.depth_data.filter(regex=side + ".+price").rename(columns=lambda x: x.split('_')[0])

    def get_size(self, side):
        """
        Obtain size levels of a particular trading side with renamed column names
        Parameters
        ----------
        side : {"bid", "ask"}
            Trading sides
        Returns
        -------
            pd.DataFrame
        """
        assert side in {"bid", "ask"}, "Trading sides should be {'bid', 'ask'}"
        return self.depth_data.filter(regex=side + ".+size").rename(columns=lambda x: x.split('_')[0])

    @property
    def true_price(self):
        """
        Calculate the volume weighed mid-price
            Q_true = (Q_b*P_b + Q_a*P_a) / (Q_a+Q_b)
        Returns
        -------
            pd.Series
        """
        return (self.depth_data["bid1_price"] * self.depth_data["bid1_size"] + self.depth_data["ask1_price"] *
                self.depth_data["ask1_size"]) / (self.depth_data["bid1_size"] + self.depth_data["ask1_size"])

    @property
    def tick_size(self, digit=2):
        """
        Obtain the minimum tick size of the order book
        Parameters
        ----------
        digit : int
            the number of digits to round up

        Returns
        -------
            float: rounded tick_size
        """
        bid_price, ask_price = self.get_price("bid"), self.get_price("ask")
        return min(abs(bid_price["bid1_price"] - bid_price["bid2_price"]).min().round(digit),
                   abs(ask_price["ask1_price"] - ask_price["ask2_price"]).min().round(digit))
    @property
    def mid_price(self):
        return ((self.depth_data["bid1_price"] + self.depth_data["ask1_price"]) / 2).rename("mid_price")
    @property
    def static_limit(self, price="mid_price"):
        """
        Obtain the time limit that defines spans of time which could be reasonably be considered instantaneous for
        the purpose of measuring the flow of traffic through a market.

        Calculation method:
            sigma_t = sigma_s * t^2
            where sigma_s is the volatility per second. Let sigma_t equals to the 0.5*tick size, we have
            t = 1/4*(tick/sigma_s)^2

        Parameters

        Returns
        -------
            float : seconds per half tick
        See Also
        --------
        Burghardt, Galen, et al. "Measuring market impact and liquidity." The Journal of Trading 1.4 (2006): 70-84.
        """

        mid_price = self.depth_data["mid_price"]
        sigma_s = mid_price.std() / ((mid_price.index[-1] - mid_price.index[0]).total_seconds()) ** 0.5

        return 0.25 * (self.tick_size / sigma_s) ** 2

    def volatility(self, side="mid", window=None):
        """
        Calculate the volatility of a given price per second

        Parameters
        ----------
        price : {"mid", "ask", "bid"}
             (the default is "mid_price", which is calculated by (best_ask+best_bid)/2 )
        window : str. Length of the rolling window, in which we calculate the volatility 

        """
        if side is "mid":
            price_data = self.mid_price
        else:
            price_data = self.get_price(side)
        price_data = price_data.pct_change().dropna()

        if not window:
            return price_data.std()*price_data.shape[0]
            #return price_data.std()
        else:
            tmpRolling = price_data.rolling(window)
            return (tmpRolling.std() * tmpRolling.count()).rename("rolling_std")
            #return (tmpRolling.std()).rename("rolling_std")


class Metrics(Analysis):
    def __init__(self, depth_data, quote_data, trade_data):
        super().__init__(depth_data, quote_data, trade_data)

    def trade_rate(self, window="5s", full_length=True):
        """
        Calculate the number of transactions in a rolling window

        Parameters
        ----------
        window : string
            the width of the rolling window
        full_length : Boolean
            if True, discard all rolling windows that are shorter than the defined value
        Returns
        -------
            trade_rate : Series
        """
        self.trade_data["trade_count"] = 1
        trades = self.trade_data["trade_count"].resample("1s", closed="right", label="right").sum()
        min_periods = int(pd.to_timedelta(window).total_seconds()) if full_length else 0
        return trades.rolling(window, min_periods).sum()

    def quote_rate(self, window="5s", full_length=True):
        """
        Return the number of quotes in a rolling window

        Parameters
        ----------
        window : string
            the width of the rolling window
        full_length : Boolean
            if True, discard all rolling windows that are shorter than the defined value
        Returns
        -------

        """
        self.quote_data["quote_count"] = 1
        quotes = self.quote_data["quote_count"].resample("1s", closed="right", label="right").sum()
        min_periods = int(pd.to_timedelta(window).total_seconds()) if full_length else 0
        return quotes.rolling(window, min_periods).sum()

    def spread(self, abs_spread=True, log_spread=False, base="mid_price"):
        """
        Return spreads calculated with different methods.
        * Absolute spread - Sabs_t = p_{t}^{A} - p_{t}^{B}
        * Relative spread - SrelM_t = (p_{t}^{A} - p_{t}^{B}) / p_{t}^{M}

        Parameters
        ----------
        abs_spread : bool
            whether to use the absolute spread
        base : {"mid_price", "last_trade"}
            base amount used in the calculation of relative spreads - mid price or the last trade price

        Returns
        -------
        """
        if abs_spread is True:
            if log_spread:
                return (np.log(self.depth_data["ask1_price"]) - np.log(self.depth_data["bid1_price"])).rename(
                    "log_spread")
            else:
                return (self.depth_data["ask1_price"] - self.depth_data["bid1_price"]).rename("absolute_spread")
        else:
            assert base in {"mid_price", "last_price"}, "Unknown input for `base`"
            return ((self.depth_data["ask1_price"] - self.depth_data["bid1_price"]) / self.depth_data[base]).rename(
                "relative_spread_{}".format(base))

    def effective_spread(self):
        """
        Calculate the effective spread.
        Seff_t = | p_t - p_{t}^{M}|
            p_t: the last traded price before time t
            p_{t}^{M}: mid price at time i

        Returns
        -------
        """
        return (abs(self.depth_data["mid_price"] - self.depth_data["last_price"])).rename("effective_spread")

    def realized_spread(self, dt="2s"):
        """Calculate the realized spread

        To quantify the liquidity risk bore by dealers or liquidity suppliers.

            S_r = d_t (p_t - m_{t+\Delta} ) = d_t ( p_t - m_t ) - d_t ( m_{t+\Delta} - m_t )
            E(S_r) = E(S_e) - E( d_t( m_{t+\Delta}-m_t ) )
            d_t: Trade direction at time t
            p_t: The execution price at time t
            m_t: The mid price at time t
            m_{t+\Delta}: The mid price at t+\Delta. Usually, \Delta is the value that how quickly
            market participants can adjust their quotes after a transaction.

        Parameters
        ----------
        dt : timedelta
            usually, `dt` is the value that how quickly market participants can adjust their quotes after a transaction.

        Returns
        -------

        """
        transaction_direction = np.where(self.trade_data["taker_side"] == "BUY", 1, -1)
        mid_price = self.depth_data["mid_price"].reindex(self.trade_data.index + pd.to_timedelta(dt),
                                                         method="ffill").drop_duplicates()
        mid_price.index = mid_price.index - pd.to_timedelta(dt)
        return transaction_direction * (self.trade_data["price"] - mid_price)

    def dollar_vol(self, window="1h", full_length=True, resample=True):
        """
        Calculate the dollar volume traded.

        Parameters
        ----------
        window : string
            the width of the rolling window
        full_length : Boolean
            if True, discard all rolling windows that are shorter than the defined value
        resample : Boolean
            if True, the time index of the trade data will be aligned to the depth data.

        Returns
        -------

        """

        min_periods = int(pd.to_timedelta(window).total_seconds()) if full_length else 0

        if not resample:
            return (self.trade_data["price"] * self.trade_data["size"]).groupby(level=0).sum().rolling(window,
                                                                                                       min_periods).sum().rename(
                "dollar_vol")
        else:
            trade_dollar = (self.trade_data["price"] * self.trade_data["size"]).resample("1s", closed="right",
                                                                                         label="right").sum()
            return trade_dollar.rolling(window, min_periods).sum().rename("dollar_vol")

    def edv(self):
        """Calculate the expected daily volume
        """
        daily_volume = self.trade_data["price"] * self.trade_data["size"]
        return daily_volume.resample("D").sum().mean()

    def epv(self, window="hr", **kwargs):
        """Calculate the expected period volume during `window`
        
        Parameters
        ----------
        window : str, length of the rolling window
        
        """
        return self.dollar_vol(window=window, **kwargs).mean().rename("epv")


    def turnover(self, window="1hr"):
        """
        Calculate the turnover rate: Tn = V / (S*P)
        	V: above
            S: outstanding stock of the asset
            P: average priceof the i trades
        Parameters
        ----------
        window : string
            the time range to calculate the traded dollar volume

        Returns
        -------
        """
        # TODO: BTC total shares required
        pass

    def depth(self, is_dollar=False):
        """
        Calculate the depth: D_t = q_{t}^{A} + q_{t}^{B}
            D_t: market depth in time t, calculated as sum of bid and ask volume in time t. 
            q_{t}^{A}: best ask volume in the order book
            q_{t}^{B}: best bid volume in the order book

        Or calculate the dollar depth: D$_t = (q_{t}^{A} * p_{t}^{A} + q_{t}^{B} * p_{t}^{B})/2
                D$_t: dollar depth, the average of the quoted bid and ask depths in currency terms
                p_{t}^{A}: best ask price at time t
                p_{t}^{B}: best bid price at time t
        Parameters
        ----------
        is_dollar : Bool
            whether to use the dollar depth (I.1.5)

        Returns
        -------

        """
        if not is_dollar:
            return (self.depth_data["bid1_size"] + self.depth_data["ask1_size"]).rename("depth")
        else:
            return ((
                            self.depth_data["bid1_price"] * self.depth_data["bid1_size"] *
                            self.depth_data["ask1_price"] * self.depth_data["ask1_size"]
                    ) / 2).rename("dollar_depth")

    def order_imbalance(self):
        """
        Calculate the order imbalance defined as
            I(t) = Total value of sell orders - total value of buy orders
        Returns
        -------
        """
        pass

    def LHH(self, time_range="1hr"):
        """
        Calculate the Hui-Heubel liquidity ratio.

        Parameters
        ----------
        time_range : string
            the time range to calculate the traded dollar volume

        Returns
        -------

        """
        # TODO: BTC total shares required
        pass

    def mec(self, long_term="5h", short_term="1h"):
        """
        MEC = Var(Rt) / (T * Var(rt))

        Var(Rt): variance of the logarithm of long-period returns
        Var(rt): variance of the logarithm of short-period returns
        T: number of short periods in each longer period

        Note: we use the resampled "last price" to calculate the return

        Parameters
        ----------
        long_term : string
            time-delta of the long-period
        short_term : string
            time-delta of the short-period

        Returns
        -------
        float : the market efficiency coefficients
        """
        T = (pd.to_timedelta(long_term) / pd.to_timedelta(short_term))
        return_rate = self.depth_data["last_price"].pct_change()
        long_period_return = return_rate.rolling(long_term,
                                                 min_periods=int(pd.to_timedelta(long_term) / pd.to_timedelta("1s"))
                                                 ).apply(lambda x: np.prod(1 + x), raw=True).dropna()

        short_period_return = return_rate.rolling(short_term,
                                                  min_periods=int(pd.to_timedelta(short_term) / pd.to_timedelta("1s"))
                                                  ).apply(lambda x: np.prod(1 + x), raw=True).dropna()

        return np.log(long_period_return).var() / (T * np.log(short_period_return).var())

    def quote_slope(self, is_log=True):
        """
        Calculate the quote slope based on:
            QS_t = Sabs_t / Dlog_t = (p_t^A-p_t^B) / (ln(q_t^A) + ln(q_t^B))

        Returns
        -------
        """
        # TODO: maybe not suitable
        if is_log:
            return ((self.depth_data["ask1_price"] - self.depth_data["bid1_price"]) / (
                    np.log(self.depth_data["ask1_size"]) + np.log(self.depth_data["bid1_size"]))).rename("quote_slope")
        else:
            return ((self.depth_data["ask1_price"] - self.depth_data["bid1_price"]) / (
                    self.depth_data["ask1_size"] + self.depth_data["bid1_size"])).rename("quote_slope_size_base")

    def log_quote_slope(self, adjusted=False):
        """
        Calculate the log quote slope based on: 
            LogQS_t = Srellog_t / Dlog_t
        Or the adjusted log quote slope:
            LogQSadj_t = LogQS_t * ( 1 + |ln( q_t^B / q_t^A )| )
        
        Parameters
        ----------
        adjusted : bool, optional
            whether to calculate the adjusted log quote slope

        Returns
        -------
        """
        log_QSt = (np.log(self.depth_data["ask1_price"]) - np.log(self.depth_data["bid1_price"])) / (
                np.log(self.depth_data["ask1_size"]) + np.log(self.depth_data["bid1_size"]))

        return log_QSt.rename("log_quote_slope") if not adjusted else (log_QSt * (
                1 + abs(np.log(self.depth_data["bid1_size"] / self.depth_data["ask1_size"])))).rename("adj_log_quote_slope")

    def composite_liquidity(self):
        """
        Calculate the composite liquidity based on:
            CL_t = SrelM_t / D$_t = (p_t^A - p_t^B) / p_t^M / ((q_t^A*p_t^A + q_t^B*p_t^B)/2)
        
        SrelM_t : relative spread with the mid price 
        D$_t: dollar depth

        Returns
        -------
        """
        return 2 * (self.depth_data["ask1_price"] - self.depth_data["bid1_price"]) / self.depth_data["mid_price"] / (
                self.depth_data["ask1_size"] * self.depth_data["ask1_price"] +
                self.depth_data["bid1_size"] * self.depth_data["bid1_price"]
        ).rename("composite_liquidity")

    def liquidity_ratio_1(self, window="1h"):
        """
        Calculate the liquidity ratio by
            LR1_t = V_t / |r_t|
            V_t = \sum_{i=1}^N p_i*q_i
            r_t = returns from period t-1 to t

        Parameters
        ----------
        window :
            time string : calculate LR3_t. `window` is the rolling window

        Returns
        -------
        """
        dollar_volume = self.dollar_vol(window=window, full_length=True, resample=True).reindex(self.depth_data.index,
                                                                                                method="ffill")

        return_rate = self.depth_data["last_price"].pct_change()
        cumulative_return_rate = return_rate.rolling(window, int(pd.to_timedelta(window).total_seconds())).apply(
            lambda x: np.prod(1 + x), raw=True)
        cumulative_return_rate[cumulative_return_rate==0] = np.nan

        return (dollar_volume / cumulative_return_rate).rename("liquidity_ratio_1")

    def liquidity_ratio_3(self, window="1h"):
        """
        Calculate the liquidity ratio by
            LR3_t = \sum_{i=1}^N |r_i| / N_t
            r_i: the price change per transaction
            N_t: the number of trades

        Parameters
        ----------
        window :
            time string : calculate LR3_t. `window` is the rolling window
        """

        return_rate = self.depth_data["last_price"].pct_change()
        cumulative_return_rate = return_rate.rolling(window).apply(lambda x: np.prod(1 + x)-1, raw=True)

        self.depth_data["count"] = 1
        trades = self.depth_data["count"].resample("1s", closed="right", label="right").sum()
        cumulative_trades = trades.rolling(window).sum()
        self.depth_data = self.depth_data.drop(["count"], axis=1)

        return (cumulative_return_rate / cumulative_trades).rename("liquidity_ratio_3").dropna()

    def flow_ratio(self, window="1h", is_wait_time=True):
        """
        Calculate the flow ratio by:
            FR_t = V_t / WT_t
            WT_t: average waiting time between trades

        Or by:
            FR_t = V_t * N_t

            V_t: the dollar volume within the window
            N_t: the number of trades within the window

        Parameters
        ----------
        window : str, optional
            the length of the rolling time window

        Returns
        -------
        """

        if is_wait_time:
            dollar_volume = self.dollar_vol(window=window, full_length=True, resample=False)
            Trade_data = self.trade_data
            Trade_data['time_exchange'] = Trade_data.index
            Trade_data['wait_time'] = (Trade_data['time_exchange'] - Trade_data['time_exchange'].shift(1)).apply(
                func=lambda row: row.total_seconds())
            wait_time = Trade_data['wait_time'].rolling(window).mean()
            return (dollar_volume / wait_time).rename("flow_ratio")

        else:
            return (self.dollar_vol(window=window, full_length=True, resample=True)
                    * self.trade_rate(window=window, full_length=True)).rename("flow_ratio")


    def order_ratio(self, window="1h"):
        """
        Compare the order ratio by
            OR_t = | q_t^B - q_t^A | / V_t
            V_t: turnover, V_t = \sum_{i=1}^{N_t} p_i * q_i

        Parameters
        ----------
        window : str, optional
            the length of the rolling time window

        Returns
        -------
        """
        spreads = abs(self.depth_data["ask1_size"] - self.depth_data["bid1_size"])
        turnovers = self.dollar_vol(window, full_length=True, resample=True)
        # turnovers = (self.trade_data["price"] * self.trade_data["size"]).groupby(level=0).sum().rolling(
        #     window, min_periods=int(pd.to_timedelta(window).total_seconds())).sum().reindex(self.depth_data.index,
        #                                                                                     method="ffill")
        return (spreads / turnovers).rename("order_ratio_{}".format(window))

    def sweep_fill_price(self, side, total_size=0.5):
        """
        Calculate the sweep to fill price, which is the volume-weighted average price to fill a certain number
        of contracts.
        Parameters
        ----------
        side : {"bid", ask"}
            Trading sides
        total_size : float
            Total number of contracts to be filled
        Returns
        -------
            pd.Series. volume-weighted average price
        """
        order_price = self.get_price(side)
        order_size = self.get_size(side)
        mask = (order_size.cumsum(axis=1) < total_size)  # Find the mask for levels that have been eaten up
        filled_order_size = order_size * mask  # Find the order sizes of levels that have been eaten up
        shifted_mask = mask.shift(1, axis=1).fillna(method="backfill", axis=1)

        # Using logical XOR to identify the new price level
        idx_best_px = np.argmax(mask.values ^ shifted_mask.values, axis=1)

        # Find # of contracts filled at the new best px level.
        size_on_book = order_size.sum(axis=1)
        total_size = np.where(size_on_book < total_size, size_on_book, total_size)
        remaining_order = (total_size - filled_order_size.sum(axis=1)).values

        filled_order_size.values[range(filled_order_size.shape[0]), idx_best_px] = remaining_order
        return (order_price * filled_order_size).sum(axis=1) / filled_order_size.sum(axis=1)

    def get_enlarged_price(self, turnover, side):
        """
        Calculate the best bid/ask price after absorbing "turnover" dollars
        Parameters
        ----------
        turnover : float
            Total amount of dollars to be absorbed
        side : {"ask", "bid"}
            Trade sides
        Returns
        -------
            pd.Series
        """
        assert side in {"ask", "bid"}, "Trade sides should be 'ask' or 'bid'."
        max_level = self.dims[1]
        price, size = self.get_price(side), self.get_size(side)
        price_labels = ((price * size).cumsum(axis=1) < turnover).sum(axis=1).map(
            lambda x: side + "{}".format(min(max_level, x + 1)))
        return pd.Series(price.lookup(price_labels.index, price_labels.values), index=price_labels.index)

    def market_impact(self, turnover=10000, side="two"):
        """ Calculate the market impact
        Enlarge the quoted spread to a certain turnover that has to be generated.
            * Two-side
            MI_t^{V*} = p_t^{A, V*} - p_t^{B, V*}
            * Single-side: maybe useful in a rapidly moving market
            MI_t^{A, V*} = p_t^{A, V*} - p_t^{M}
            MI_t^{B, V*} = p_t^{M} - p_t^{B, V*} 

        Parameters
        ----------
        turnover : float, optional
            the target turnover
        side : {"two", "bid", "ask"}
            which side to calculate the the market impact
        
        """
        assert side in {"two", "ask", "bid"}, "Market Impact sides should be 'two', 'ask', 'bid'."
        if side == "two":
            return (self.get_enlarged_price(turnover, "ask") - self.get_enlarged_price(turnover, "bid")).rename(
                "market_impact_two_side")
        else:
            return abs(self.get_enlarged_price(turnover, side) - self.depth_data["mid_price"]).rename(
                "market_impact_{}".format(side))
