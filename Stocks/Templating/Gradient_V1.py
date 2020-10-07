# gradient test
from talib import RSI, BBANDS, MACD
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def PlotMACD(shareCode, showChart=False):
    df = pd.read_json('D:\\Personal\\share\\data\\' + shareCode + '.txt')
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df['t'] = df['t'].dt.date
    df.set_index('t', inplace = True)

    df.drop('s', axis=1, inplace=True)

    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 7
    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    df['MACD'] = macd
    df['MACDS'] = macdSignal
    df['MACDH'] = macdHist
    df['MACDHDIFF'] = df['MACDH'].diff()

    xaxisType = {'type':"category", 'categoryorder':'category ascending'}
    yaxisType = {'autorange': True, 'fixedrange':True}

    layout = dict( title=shareCode,
                xaxis = xaxisType,
                xaxis2 = xaxisType,
                xaxis3 = xaxisType,
                yaxis = yaxisType)

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    #fig.append_trace(go.Candlestick(x=df.index, open=df.o, close=df.c, high=df.h, low=df.l), row=1, col=1)

    macdRow = 1
    fig.append_trace(go.Scatter(x=df.index, y=df['MACDHDIFF'], mode='lines', name='Gradient'), row=macdRow, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line={'color':'blue'}), row=macdRow, col=1)
    fig.append_trace(go.Scatter(x=df.index, y=df['MACDS'], mode='lines', name='MACD SIGNAL', line={'color':'red'}), row=macdRow, col=1)
    fig.append_trace(go.Bar(x=df.index, y=df['MACDH'], name='MACD Hist'), row=macdRow, col=1)

    fig.update_layout(layout)
    if showChart:
        fig.show()
    else:
        fig.write_html('D:\\Personal\\share\\results\\charts\\' + shareCode + '.html', include_plotlyjs='cdn')
    return

PlotMACD('RCL.N0000', True)