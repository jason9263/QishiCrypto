import ccxt
from pprint import pprint
import mysql.connector

# [
#     1504541580000, // UTC timestamp in milliseconds, integer
#     4235.4,        // (O)pen price, float
#     4240.6,        // (H)ighest price, float
#     4230.0,        // (L)owest price, float
#     4230.7,        // (C)losing price, float
#     37.72941911    // (V)olume (in terms of the base currency), float
# ],

#huobipro = ccxt.huobipro()

huobipro = ccxt.binance()

datestamp = ['2018-01-01T00:00:00.000Z', '2018-01-01T12:29:00.000Z',
             '2018-04-01T00:00:00.000Z', '2018-04-01T12:29:00.000Z',
             '2018-07-01T00:00:00.000Z', '2018-07-01T12:29:00.000Z',
             '2018-10-01T00:00:00.000Z', '2018-10-01T12:29:00.000Z']

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="bmxss007",
    database="qishi"
)

print(mydb)

mycursor = mydb.cursor()

cnt = 1

for index in datestamp:
    ohlcv = huobipro.fetch_ohlcv('ETH/USDT', '1m', since=huobipro.parse8601(index), limit=500)
    # print(index)
    for candle in ohlcv:
        print([huobipro.iso8601(candle[0])] + candle[1:])

        sql = "INSERT INTO binanceohlcv (id,timestamp,open_price, highest_price,lowest_price, closing_price, volume) VALUES (%s ,%s , %s , %s,%s,%s,%s )"
        val = (cnt, candle[0], candle[1], candle[2], candle[3], candle[4], candle[5])
        mycursor.execute(sql, val)
        cnt = cnt + 1

        mydb.commit()

        print(mycursor.rowcount, "record inserted.")

mydb.close()
