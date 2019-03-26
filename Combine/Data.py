import pandas as pd
from re import findall


class Feeder:
    def __init__(self, data_path=None):
        """
        Parameters
        ----------
        data_path : string, required
            The path of the data files.
        
        """

        try:
            assert data_path is not None, "Data path is None!"
        except AssertionError:
            print("Please input the Data path")

        self._Data_Path = data_path
        self.symbol = findall(r"(.+)_(depth|trade|quote).+", self._Data_Path.split("/")[-1])[0][0]


    def get(self, is_raw=False):
        if is_raw:
            return self._get_raw()
        else:
            return self._get_clean()

    def _get_raw(self):
        self.date = findall(r"(\d{8})", self._Data_Path)[0]
        market_data = pd.read_csv(self._Data_Path, delimiter=";")
        market_data["symbol"] = self.symbol
        market_data.time_exchange = pd.to_datetime(self.date, format="%m%d%Y") + pd.to_timedelta(market_data.time_exchange)
        market_data.time_coinapi = pd.to_datetime(self.date, format="%m%d%Y") + pd.to_timedelta(market_data.time_coinapi)
        return market_data

    def _get_clean(self):
        data = pd.read_csv(self._Data_Path, delimiter=",")
        data["time_exchange"] = pd.to_datetime(data["time_exchange"])
        data.set_index("time_exchange", inplace=True)
        return data