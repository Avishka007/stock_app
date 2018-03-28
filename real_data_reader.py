import csv 
import numpy as np
from statistics import mean
from collections import defaultdict


from predicted_scv_slicer import m1, m2, m3, m4, m5, m6, m7

columns = defaultdict(list)

with open('realdata.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
col1 = (columns[1])
col = [float(i) for i in col1]
#print (col[0:10])

#first real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[0:49])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m11 = (m*100)


#2nd real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[58:107])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m22 = (m*100)


#3rd real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[116:165])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m33 = (m*100)


#4th real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[174:223])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m44 = (m*100)


#5th real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[232:281])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m55 = (m*100)


#6th real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[290:339])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m66 = (m*100)


#7th real data slice
xs = np.array(list(range(0, 49)))
ys = np.array(col[348:397])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)
m77 = (m*100)


a2 = m2/m22
a2 = float("{0:.2f}".format(a2))
a3 = m3/m33
a3 = float("{0:.2f}".format(a3))
a4 = m4/m44
a4 = float("{0:.2f}".format(a4))
a5 = m5/m55
a5 = float("{0:.2f}".format(a5))
a6 = m6/m66
a6 = float("{0:.2f}".format(a6))
a7 = m7/m77
a7 = float("{0:.2f}".format(a7))

q = [a2,a3,a4,a5,a6,a7]
result = float("{0:.2f}".format(np.prod(np.array(q))))
print (q)





	
		

