import pandas_datareader.data as web
import datetime
import pandas as pd
import csv


stock = "AAPL"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

print(df.Close.head())





dk = pd.DataFrame(df.Close)
dk.to_csv("slope.csv")
