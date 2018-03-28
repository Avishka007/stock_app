import pandas_datareader.data as web
import datetime
import pandas as pd
import csv


stock = "AAPL"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/AAPL.csv")

stock = "GOOG"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/GOOG.csv")

stock = "AAPL"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/AAPL.csv")

stock = "EBAY"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/EBAY.csv")

stock = "dowjhones"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/dowjhones.csv")

stock = "AMZON"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/AMZON.csv")

stock = "KO"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/KO.csv")

stock = "CITI"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/CITI.csv")

stock = "TATA"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/TATA.csv")

stock = "yahoo"
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()
df = web.DataReader(stock, 'google', start, end)

dk = pd.DataFrame(df.Close)
dk.to_csv("./stock/yahoo.csv")

