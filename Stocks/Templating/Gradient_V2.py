import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from talib import RSI, BBANDS, MACD

def testFunc(shareCode):
    df = pd.read_json('D:\\Personal\\share\\data\\' + shareCode + '.txt')
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df['t'] = df['t'].dt.date
    df.set_index(pd.DatetimeIndex(df['t']), inplace = True, drop=False)
    
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 8
    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    

    df['macd'] = macd
    df['macds'] = macdSignal
    df['macdh'] = macdHist

    df['diff'] = df['macdh'].diff()
    df['ema'] = df['c'].rolling(window=21).mean()

    slope = pd.Series(np.gradient(df['ema']), df.index, name='slop')
    df['slop'] = slope
    #print(df)
    
    xaxisType = {'type':"category", 'categoryorder':'category ascending'}
    yaxisType = {'autorange': True, 'fixedrange':True}

    layout = dict( title=shareCode,
                xaxis = xaxisType,
                xaxis2 = xaxisType,
                xaxis3 = xaxisType,
                yaxis = yaxisType)
    
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    #fig.append_trace(go.Candlestick(x=df.index, open=df.o, close=df.c, high=df.h, low=df.l), row=1, col=1)

    macdRow =1
    #fig.append_trace(go.Scatter(x=df.index, y=df['MACDHDIFF'], mode='lines', name='Gradient'), row=macdRow, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD'), row=macdRow, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['macds'], mode='lines', name='MACD SIGNAL'), row=macdRow, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['diff'], mode='lines', name='Diff'), row=macdRow, col=1)
    #fig.append_trace(go.Scatter(x=df.index, y=df['ema'], mode='lines', name='Ema'), row=macdRow, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['slop'], mode='lines', name='Slope'), row=macdRow, col=1)
    fig.append_trace(go.Bar(x=df.index, y=df['macdh'], name='MACD Hist'), row=macdRow, col=1)

    fig.update_layout(layout)
    fig.show()

    #### Using Express
    # fig = px.line(df, x='t', y=['diff', 'slop'], title=shareCode)
    # fig.add_bar(x=df['t'], y=df['macdh'])

    # xaxisType = {'type':"category", 'categoryorder':'category ascending'}
    # layout = dict(xaxis = xaxisType)
    # fig.update_layout(layout)
    # fig.show()

testFunc('NDB.N0000')

    
#dfChart=df.resample('1B').asfreq()
#df.reindex(pd.date_range(start=df['t'].min(), end=df['t'].max(), freq='D'))
#print(dfChart.index)
#test = pd.date_range(start=df['t'].min(), end=df['t'].max(), freq='D')
#print(test)
