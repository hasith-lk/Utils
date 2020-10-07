# Using graph_objects
import plotly.graph_objects as go
import datetime

import pandas as pd
df = pd.read_json('D:\\Personal\\share\\data\\JKH.N0000.txt')
df['t'] = pd.to_datetime(df['t'],unit='s')
df['date'] = df['t'].dt.date
df.set_index('date', inplace = True)
df.head()

# First instance
trace1 = {
    'x': df.index,
    'open': df.Open,
    'close': df.Close,
    'high': df.High,
    'low': df.Low,
    'type': 'candlestick',
    'name': 'JKH',
    'showlegend': False
}

data =[go.Candlestick(x=df.index, 
    open = df['o'],
    high = df['h'],
    low = df['l'],
    close = df['c'],
    name = df['date'])]

layout = dict(
            title="JKH.N0000",
            xaxis = dict(
            type="category",
            categoryorder='category ascending',
            tickformat='%Y %m %d'))

fig = go.Figure(data=data, layout=layout)

fig.show()


# day_delta = datetime.timedelta(1,0,0)
# next = df['t'][0]
# print(next)
# print(df['t'][-1] )
# print((df['t'][-1] - next))

# days = []
# while (df['t'][-1] - next) > day_delta:
#     next = next + day_delta
#     days.append(next)

# print(days)