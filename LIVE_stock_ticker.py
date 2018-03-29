import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt

app = dash.Dash()

app.layout = html.Div([
    html.H1('Select your Stock'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Amazon', 'value': 'AMZON'},
            {'label': 'Google', 'value': 'GOOG'},
            {'label': 'Apple', 'value': 'AAPL'},
            {'label': 'Dow johns', 'value': 'DOWJ'},
            {'label': 'eBay', 'value': 'EBAY'},
            {'label': 'Apple', 'value': 'AAPL'},
            {'label': 'TATA', 'value': 'TATA'},
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Yahoo', 'value': 'YAHO'},
            {'label': 'Toyota', 'value': 'TOYA'}
        ],
        value='COKE'
    ),
    dcc.Graph(id='my-graph')
])

@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    df = web.DataReader(
        selected_dropdown_value, data_source='google',
        start=dt(2017, 1, 1), end=dt.now())
    return {
        'data': [{
            'x': df.index,
            'y': df.Close
        }]
    }

if __name__ == '__main__':
    app.run_server(port = 9090, debug=True)