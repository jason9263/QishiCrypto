import ccxt
import time
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib import style

now = lambda: time.time()
start = now()

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def parseData(exchange, symbol):
    # fetch data from exchange
    tickerInfo = exchange.fetch_ticker(symbol)
    exchange_depth = exchange.fetch_order_book(symbol)


    # prepare output flow - depth
    output = {}
    output["time_exchange"] = tickerInfo["datetime"]
    output["symbol_id"] = "BINANCE_SPOT_BTC_USDT"
    output["price"] = tickerInfo["info"]["lastPrice"]
    output["size"] = tickerInfo["info"]["lastQty"]
    for idx, bid_level in enumerate(exchange_depth.get('bids')[:20], 1):
            output["bid{}_price".format(idx)] = bid_level[0]
            output["bid{}_size".format(idx)] = bid_level[1]
    for idx, ask_level in enumerate(exchange_depth.get('asks')[:20], 1):
            output["ask{}_price".format(idx)] = ask_level[0]
            output["ask{}_size".format(idx)] = ask_level[1]
    print("time:", output["time_exchange"], "\t price:", output["price"],"\t size:", output["size"])

    data = pd.DataFrame(output, index=[0])
    data["time_exchange"] = pd.to_datetime(data["time_exchange"])
    return data


def parseTrade(exchange, symbol, limit=100):
    exchange_trades = exchange.fetch_trades(symbol, limit=limit)
    for item in exchange_trades:
        item.pop("info", None)
    data = pd.concat(map(lambda x: pd.DataFrame(x, index=[0]).set_index('id'), exchange_trades))
    data["datetime"] = pd.to_datetime(data["datetime"])
    data.rename(columns={'datetime':'time_exchange','amount':'size'}, inplace=True)
    return data

def main():
    buffer = 1
    limit = 360000*24

    order_book_cols = ["time_exchange","symbol_id","price","size"]
    for x in range(1,21):
        order_book_cols.extend(["bid{0:}_price".format(x),"bid{0}_size".format(x),
                        "ask{0:}_price".format(x),"ask{0}_size".format(x)])
    order_book = pd.DataFrame(columns=order_book_cols)

    trade_book = pd.DataFrame(columns=['timestamp', 'datetime', 
                                        'symbol', 'id', 
                                        'order', 'type', 
                                        'takerOrMaker', 'side', 
                                        'price', 'cost', 'amount', 'fee'])

    cnt = 1
    # with open(, 'w') as f:
    while True:
        time.sleep(1)
        exchanges = [ccxt.binance()]
        symbols = ['BTC/USDT','BTC/USTD']
        for i in range(len(exchanges)):
            exchange = exchanges[i]
            symbol = symbols[i]
            order_book = order_book.append(parseData(exchange, symbol), sort=False)
            trade_book = trade_book.append(parseTrade(exchange, symbol), sort=False).drop_duplicates()
            cnt = cnt + 1
            if cnt%buffer==0:
                print("writing...",end="")
                order_book.tail(limit).to_csv('./btc_usdt_depth.csv', index=False, header=True)
                order_book.tail(limit).to_pickle("./btc_usdt_depth.pkl")
                trade_book.to_csv('./btc_usdt_trade.csv', header=True)
                trade_book.to_pickle("./btc_usdt_trade.pkl")
                print("done!")

if __name__ == '__main__':
    main()
    print('Run Time: %s' % (now() - start))
