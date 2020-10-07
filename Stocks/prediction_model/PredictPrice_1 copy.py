import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, MACD

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('ggplot')

stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

shareCode='COMB.N0000'

df = pd.read_json(dataFolderPath+'\\' + shareCode + '.txt')

MACD_FAST = 10
MACD_SLOW = 21
MACD_SIGNAL = 8

macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

df['macd'] = macd
df = df[['macd']]
df = df.dropna()
#df.shape

#Create a variable to predict 'x' days out into the future
future_days = 20

#Create a new column (the target or dependent variable) shifted 'x' units/days up
df['Prediction'] = df[['macd']].shift(-future_days)
#print the data
df.shape

# feature dataset
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(X)

# target data set
y = np.array(df['Prediction'])[:-future_days]
print(y)


X = StandardScaler().fit_transform(X)

# split dataset to train set and test set (25%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Create the decision tree regressor model
# tree = DecisionTreeRegressor().fit(x_train, y_train)
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

#Show the model tree prediction
#tree_prediction = tree.predict(x_future)
#print( tree_prediction )
#print()
#Show the model linear regression prediction
y_prediction = lr.predict(x_test)
print(y_prediction)

lrConfus = confusion_matrix(y_test, y_prediction)

print("{} - {} - {} - {} - {} - {} - {}".format(shareCode, lr.score(x_train, y_train), lr.score(x_test, y_test), lrConfus[0,0], lrConfus[0,1], lrConfus[1,0], lrConfus[1,1]))

# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
# export_graphviz(tree, out_file = (resultFolderPath + '\\tree.dot'), feature_names =['MACD'])

# #Visualization
# predictions = tree_prediction

# #Plot the data
# valid =  df[X.shape[0]:]
# valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices
# plt.figure(figsize=(16,8))
# plt.title('Tree Model')
# plt.xlabel('Days',fontsize=18)
# plt.ylabel('Close Price LKR (R)',fontsize=18)
# plt.plot(df['macd'])
# plt.plot(valid[['macd','Predictions']])
# plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
# plt.show()

#Visualize the data
predictions = lr_prediction
#Plot the data
valid =  df[X.shape[0]:]
valid['Predictions'] = predictions #Create a new column called 'Predictions' that will hold the predicted prices
plt.figure(figsize=(16,8))
plt.title('Regression Model')
plt.xlabel('Days',fontsize=18)
plt.ylabel('Close Price LKR (R)',fontsize=18)
plt.plot(df['macd'])
plt.plot(valid[['macd','Predictions']])
plt.legend(['Train', 'Val', 'Prediction' ], loc='lower right')
plt.show()
