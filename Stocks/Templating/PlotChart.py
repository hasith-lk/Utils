# Using graph_objects
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from talib import RSI, BBANDS, MACD
import pandas as pd

def PlotStockChart(shareCode, showChart=False):

    RSI_T = 10
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 5
    BBAND_PERIOD = 21

    data = pd.read_json('D:\\Personal\\share\\data\\'+shareCode+'.txt')
    data['t'] = pd.to_datetime(data['t'],unit='s')
    data['date'] = data['t'].dt.date
    data.set_index(['date'], inplace = True, drop=False)

    df = data

    # First instance
    trace1 = {
        'x': df.index,
        'open': df.o,
        'close': df.c,
        'high': df.h,
        'low': df.l,
        'type': 'candlestick',
        'name': shareCode,
        'showlegend': False
    }

    rsi = RSI(df.c, timeperiod=10)
    macd, macdSignal, macdHist = MACD(df.c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    upper, middle, lower = BBANDS(df.c, timeperiod=BBAND_PERIOD, nbdevup=2, nbdevdn=2)

    # Second instance - avg_30
    trace2 = {
        'x': df.index,
        'y': rsi,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'blue'
                },
        'name' : 'RSI'
    }

    rsiLower = {
        'x': df.index,
        'y': 30*(rsi/rsi),
        'type': 'scatter',
        'line': {
            'width': 1,
            'color': 'red'
                },
        'name' : 'RSI Lower'
    }

    rsiUpper= {
        'x': df.index,
        'y': 70*(rsi/rsi),
        'type': 'scatter',
        'line': {
            'width': 1,
            'color': 'red'
                },
        'name' : 'RSI Upper'
    }

    # MACD
    trace4 = {
        'x': df.index,
        'y': macd,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'blue'
        },
        'name' : 'macd'
    }

    trace5 = {
        'x': df.index,
        'y': macdHist,
        'type': 'bar',
        'name' : 'macdHist'
    }

    trace6 = {
        'x': df.index,
        'y': macdSignal,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'red'
        },
        'name' : 'macdSignal'
    }

    bbUpper = {
        'x': df.index,
        'y': upper,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'blue'
        },
        'name' : 'BB Upper'
    }

    bbLower = {
        'x': df.index,
        'y': lower,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'blue'
        },
        'name' : 'BB Lower'
    }

    bbMiddle = {
        'x': df.index,
        'y': middle,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'orange'
        },
        'name' : 'BB Middle'
    }
    # Aggregate all instances and define 'data' variable

    xaxisType = {'type':"category", 'categoryorder':'category ascending'}
    layout = dict( title=shareCode,
                xaxis = xaxisType,
                xaxis3 = xaxisType,
                xaxis4 = xaxisType)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    fig.append_trace(trace1, row=1, col=1)
    fig.append_trace(bbLower, row=1, col=1)
    fig.append_trace(bbMiddle, row=1, col=1)
    fig.append_trace(bbUpper, row=1, col=1)

    fig.append_trace(trace2, row=4, col=1)
    fig.append_trace(rsiLower, row=4, col=1)
    fig.append_trace(rsiUpper, row=4, col=1)

    fig.append_trace(trace4, row=3, col=1)
    fig.append_trace(trace5, row=3, col=1)
    fig.append_trace(trace6, row=3, col=1)

    fig.update_layout(layout)

    def zoom(layout, xrange):
        in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
        fig.layout.yaxis.range = [in_view.High.min() - 10, in_view.High.max() + 10]

    #fig.layout.on_change(zoom, 'xaxis.range')

    fig.layout.on_change(lambda obj, xrange, yrange: print("%s-%s" % (xrange, yrange)),('xaxis', 'range'), ('yaxis', 'range'))

    if showChart:
        fig.show()
    else:
        fig.write_html('D:\\Personal\\share\\results\\charts\\' + shareCode + '.html', include_plotlyjs='cdn')
    return

#PlotStockChart('JKH.N0000', True)