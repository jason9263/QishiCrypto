import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self, analysis):
        self.analysis = analysis
        self.market_impact = None
        self.base_model = None

        self.data = None
        self.train_data = None
        self.test_data = None
        self.X_label = None
        self.y_label = None
    def get_market_impact(self, turnovers):
        """Calculate the market impact for different turnovers

        Parameters
        ----------
        turnovers : iterator, list of different turnovers
        """
        self.market_impact = pd.concat(map(lambda to: self.analysis.market_impact(turnover=to, side="two")/self.analysis.mid_price, turnovers), 
                                keys=turnovers, axis=1
                                )
        return self.market_impact

    def _unstack_data(self):
        try:
            return self.market_impact.unstack().reset_index().rename(
                columns={
                "level_0":"V",
                "time_exchange":"time_exchange",
                0:"MI"
                    })
        except KeyError:
            print("Make sure the market impact is correctly calculated")

    def plot(self):
        fig, ax = plt.subplots(figsize=(10,8))
        self.market_impact.mean().plot(ax=ax, marker='.')
        ax.fill_between(self.market_impact.mean().index, 
                        self.market_impact.mean() - self.market_impact.std(), 
                        self.market_impact.mean() + self.market_impact.std(),
                    alpha=0.4)
        ax.set_xlabel("Volume $")
        ax.set_ylabel("Market Impact")
        ax.set_title("Market Impact VS Order Volume")
        return ax

    def add_features(self, other_features, on="time_exchange"):
        """ Add features to the mareket impact data.
        
        Parameters
        ----------
        features : pd.Dataframe
        """
        self.data = self.data.join(other_features, on=on).dropna()

    def init(self):
        """Prepare data"""
        raise NotImplementedError

    def train(self, base_model, **kwargs):
        """Train the model
        
        Parameters
        ----------
        base_model : machine learning model with sklearn apis
        **kwargs : key word arguments that will be forwarded to the base model.

        x_label : string or list of string
        y_label : string

        # TODO: Consider to include the CV 
        """
        # --- Train test split --- #
        self.base_model = base_model
        print("Training...", end="")
        self.base_model.fit(self.data[self.X_label], self.data[self.y_label])
        print("Done!")

    def predict(self, data):
        return self.base_model.predict(data)


class Bloomberg(Model):
    def init(self, turnovers=range(500,100001,1000), window="1h"):
        # --- Preprocessing data --- #      
        _ = self.get_market_impact(turnovers)
        self.data = self._unstack_data()
        # --- Make sure `other_features` are combined as a pd.DataFrame
        other_features = pd.concat([
            self.analysis.spread(abs_spread=False),  # Relative spread
            self.analysis.volatility(window=window), # volatility of the past 1 hour
        ], axis=1
        )
        self.add_features(other_features)
        self.data["sqrt_vol_ratio"] = (self.data["V"] / self.analysis.edv())**0.5
        self.data["std_sqrt_vol_ratio"] = self.analysis.volatility() * self.data["sqrt_vol_ratio"]
        # --- feature labels and the target labels added here --- #
        self.X_label = ['V', 'relative_spread_mid_price',
                       'rolling_std', 'sqrt_vol_ratio', 'std_sqrt_vol_ratio']
        self.y_label = "MI"

class JPM(Model):
    def init(self, turnovers=range(500,5001,500), window="1h"):
        # --- Preprocessing data --- #      
        _ = self.get_market_impact(turnovers)
        self.data = self._unstack_data()
        other_features = pd.concat([
            self.analysis.spread(abs_spread=False),  # Relative spread
            self.analysis.volatility(window=window), # volatility of the past 1 hour
        ], axis=1
        )
        self.add_features(other_features)
        self.data["sqrt_v_ratio"] = (self.data["V"] / self.analysis.edv())  
        self.data["std_sqrt_vol_ratio"] = self.analysis.volatility() * self.data["sqrt_v_ratio"]

        one_hr_volume = self.analysis.dollar_vol(window='1h', full_length=True, resample=True).mean()
        self.data["temp_impact"] = self.data["temp_impact"] = (self.data["V"] / one_hr_volume) * self.data["std_sqrt_vol_ratio"]
