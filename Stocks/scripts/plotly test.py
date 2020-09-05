# Using graph_objects
import plotly.graph_objects as go

import pandas as pd
df = pd.read_json('D:\\Personal\\share\\data\\JKH.N0000.txt')
df['t'] = pd.to_datetime(df['t'],unit='s')
df['date'] = df['t'].dt.date
df.head()


data =[go.Candlestick(x=df['date'], 
    open =df['o'],
    high = df['h'],
    low=df['l'],
    close=df['c'])]

layout = dict(
            title="JKH.N0000",
            xaxis = dict(
            type="category",
            categoryorder='category ascending',
            tickformat='%Y %m %d'))

fig = go.Figure(data=data, layout=layout)

fig.show()
