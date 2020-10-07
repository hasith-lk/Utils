import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, MACD

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

plt.style.use('ggplot')

stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

shareCode='COMB.N0000'

def TestModel(days_):
    df = pd.read_json(dataFolderPath+'\\' + shareCode + '.txt')

    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 8

    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    df.tail()

    df['macd'] = macd
    df = df [['macd']]
    #df = df [['c']]
    df = df.dropna()
    #Create a variable to predict 'x' days out into the future
    future_days = days_

    #Create a new column (the target or dependent variable) shifted 'x' units/days up
    df['Prediction'] = df[['macd']].shift(-future_days)
    #print the data
    df.tail(4)

    # feature dataset
    X = np.array(df.drop(['Prediction'], 1))[:-future_days]
    #X = StandardScaler().fit_transform(X)
    #print(X)

    # target data set
    y = np.array(df['Prediction'])[:-future_days]
    #print(y)

    # split dataset to train set and test set (25%)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #Create the decision tree regressor model
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    #Create the linear regression model
    lr = LinearRegression().fit(x_train, y_train)

    #Get the feature data, 
    #AKA all the rows from the original data set except the last 'x' days
    x_future = df.drop(['Prediction'], 1)[:-future_days]
    #Get the last 'x' rows
    x_future = x_future.tail(future_days) 
    #Convert the data set into a numpy array
    x_future = np.array(x_future)
    x_future

    y_actual = df['macd'][-future_days:]
    #Show the model tree prediction
    tree_prediction = tree.predict(x_future)
    #print( tree_prediction )
    #print()
    #Show the model linear regression prediction
    lr_prediction = lr.predict(x_future)

    rms = mean_squared_error(y_actual, lr_prediction)


    #Visualization
    #predictions = lr_prediction
    predictions = tree_prediction

    #Plot the data
    valid =  df[X.shape[0]:]
    valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices
    
    plt.figure(figsize=(16,8))
    plt.title('Linear Model {} Days'.format(future_days))
    plt.xlabel('Days',fontsize=18)
    plt.ylabel('macd Value',fontsize=18)
    plt.plot(df['macd'])
    plt.plot(valid[['macd','Predictions']])
    plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
    plt.show()

    return [future_days, shareCode, lr.score(x_train, y_train)*100, lr.score(x_test, y_test)*100, rms]

results = []
for x in range(10,21):
    results.append(TestModel(x))

pdResults = pd.DataFrame(results)
pdResults.head(20)
