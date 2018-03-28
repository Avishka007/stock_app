import csv
import pandas as pd
import numpy as np
from statistics import mean


path = 'p_values.csv'
lines = [line for line in open (path)]

r1 = lines[1].strip().split(',')
rr1 =  (r1[1:51])
rr1 = [float(i) for i in rr1]
##print (rr1)
df1 = pd.DataFrame(rr1)
#df1.to_csv("1st.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr1)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 1")
m1 = m*10000


r2 = lines[2].strip().split(',')
rr2 =  (r2[1:51])
rr2 = [float(i) for i in rr2]
##print (rr2)
df2 = pd.DataFrame(rr2)
#df2.to_csv("2nd.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr2)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 2")
m2 = m*10000

r3 = lines[3].strip().split(',')
rr3 =  (r3[1:51])
rr3 = [float(i) for i in rr3]
##print (rr3)
df3 = pd.DataFrame(rr3)
#df3.to_csv("3rd.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr3)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 3")
m3 = m*10000

r4 = lines[4].strip().split(',')
rr4 =  (r4[1:51])
rr4 = [float(i) for i in rr4]
##print (rr4)
df4 = pd.DataFrame(rr4)
#df4.to_csv("4th.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr4)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 4")
m4 = m*10000

r5 = lines[5].strip().split(',')
rr5 =  (r5[1:51])
rr5 = [float(i) for i in rr5]
##print (rr5)
df5 = pd.DataFrame(rr5)
#df5.to_csv("5th.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr5)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 5")
m5 = m*10000

r6 = lines[6].strip().split(',')
rr6 =  (r6[1:51])
rr6 = [float(i) for i in rr6]
##print (rr6)
df6 = pd.DataFrame(rr6)
#df6.to_csv("6th.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr6)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 6")
m6 = m*10000


r7 = lines[7].strip().split(',')
rr7 =  (r7[1:51])
rr7 = [float(i) for i in rr7]
##print (rr7)
df7 = pd.DataFrame(rr7)
#df7.to_csv("7th.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr7)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of slope 7")
m7 = m*10000


r8 = lines[8].strip().split(',')
rr8 =  (r8[1:51])
rr8 = [float(i) for i in rr8]
##print (rr8)
df8 = pd.DataFrame(rr8)
#df8.to_csv("./final_prediction/EBAY.csv")
xs = np.array(list(range(0, 50)))
ys = np.array(rr8)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
#print(m*10000,"\t: is the m value of final prediction")
#print (rr8)


