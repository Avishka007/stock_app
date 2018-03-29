import dash
import dash_core_components as dcc
import dash_html_components as html
from statistics import mean
import numpy as np
import csv
from collections import defaultdict

columns = defaultdict(list)

with open('AAPL.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)
col1 = (columns[1])
col = [float(i) for i in col1]



xs = np.array(list(range(0, 50)))
ys = np.array(col)
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)* (mean(xs)) - mean(xs**2))))
    return m
m = best_fit_slope(xs,ys)

if m>0:
    conf=("For 1 year ahead of time - stock price is going to go UP")
else:
    conf=("For 1 year ahead of time - stock price is going to go DOWN")




app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Final Prediction '),
    dcc.Graph(
        id='example',
        figure={
            'data': [
                {'x': list(range(0, 50)), 'y': col, 'type': 'line', 'name': 'future stock movement'}
                


            ],
            'layout': {
                'title':  conf
                
            }
        }
    )
])



if __name__ == '__main__':
    app.run_server(port = 8082, debug=True)



