import datetime                                                                 
import pandas_datareader as pdr                                                 
                                                                                
start = datetime.datetime(2005, 1, 1)                                           
end = datetime.datetime(2018, 02, 02)                                           
                                                                                
                                                                                
bolt = pdr.get_data_yahoo('AAPL', start=start, end=end)                         
                                                                                
print(bolt.tail()) 