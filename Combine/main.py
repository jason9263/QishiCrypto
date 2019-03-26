import sys
sys.path.append("/Users/z_runmin/Dropbox/Qishi/Code")
sys.path.append("/home/runmin/Dropbox/Qishi/Code")
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Utils.Metrics import Metrics
from Utils.Model import Bloomberg

def fetcher(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    fig, ax = plt.subplots()
    while True:
        time.sleep(1)
        Depth_data = fetcher("../Combine/btc_usdt_depth.pkl").set_index("time_exchange")
        Trade_data = fetcher("../Combine/btc_usdt_trade.pkl").reset_index().set_index("time_exchange")
        Quote_data = None

        Analysis = Metrics(Depth_data, Quote_data, Trade_data)
        test = Bloomberg(Analysis)
        test.init()
        back_bone = LinearRegression(fit_intercept=False,n_jobs=6, normalize=True)
        test.train(base_model=back_bone)

        test.data["Predict"] = test.predict(test.data[test.X_label])
        test.data[["V","MI","Predict"]].groupby(by="V").mean().plot(ax=ax)
        print(test.data[["V","MI","Predict"]].groupby(by="V").mean())

if __name__ == '__main__':
    main()
