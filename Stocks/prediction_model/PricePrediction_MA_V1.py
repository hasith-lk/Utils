import math
import numpy as np
import pandas as pd
from datetime import date, datetime, time, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from talib import RSI, BBANDS, MACD

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')

stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

shareCode='VONE.N0000'

def get_preds_mov_avg(df, target_col, N, pred_min, offset):
    """
    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
    Using simple moving average.
    Inputs
        df         : dataframe with the values you want to predict. Can be of any length.
        target_col : name of the column you want to predict e.g. 'adj_close'
        N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
        pred_min   : all predictions should be >= pred_min
        offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
    Outputs
        pred_list  : list. The predictions for target_col. np.array of length len(df)-offset.
    """
    pred_list = df[target_col].rolling(window = N, min_periods=1).mean() # len(pred_list) = len(df)
    
    # Add one timestep to the predictions
    pred_list = np.concatenate((np.array([np.nan]), np.array(pred_list[:-1])))
    
    # If the values are < pred_min, set it to be pred_min
    pred_list = np.array(pred_list)
    pred_list[pred_list < pred_min] = pred_min
    
    return pred_list[offset:]

# Prepare Data
df = pd.read_json(dataFolderPath+'\\' + shareCode + '.txt')

df['t'] = pd.to_datetime(df['t'], unit='s')
df['t'] = df['t'].dt.date

df.loc[:, 'date'] = pd.to_datetime(df['t'],format='%Y-%m-%d')

# Get month of each sample
df['month'] = df['date'].dt.month

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

df.tail(10)

ax = df.plot(x='date', y='c', style='b-', grid=True)
ax.set_xlabel("date")
ax.set_ylabel("LKR")
#plt.show()

test_size = 0.2                 # proportion of dataset to be used as test set
cv_size = 0.2                   # proportion of dataset to be used as cross-validation set
Nmax = 15                       # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
                                # Nmax is the maximum N we are going to test

# Get sizes of each of the datasets
num_cv = int(cv_size*len(df))
num_test = int(test_size*len(df))
num_train = len(df) - num_cv - num_test

# Split into train, cv, and test
train = df[:num_train]
cv = df[num_train:num_train+num_cv]
train_cv = df[:num_train+num_cv]
test = df[num_train+num_cv:]

# Plot Each data set (train, validation, test)
ax = train.plot(x='date', y='c', style='b-', grid=True)
ax = cv.plot(x='date', y='c', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='c', style='g-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test'])
ax.set_xlabel("date")
ax.set_ylabel("LKR")
plt.title("Original Data Set")
plt.show()

# Function to calculate mape value
def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

RMSE = []
mape = []
# Run from 1 to Nmax to identify best N
for N in range(1, Nmax+1): # N is no. of samples to use to predict the next value
    est_list = get_preds_mov_avg(train_cv, 'c', N, 0, num_train)
    
    cv.loc[:, 'est' + '_N' + str(N)] = est_list
    RMSE.append(math.sqrt(mean_squared_error(est_list, cv['c'])))
    mape.append(get_mape(cv['c'], est_list))
print('RMSE = ' + str(RMSE))
print('MAPE = ' + str(mape))
#df.head()

# Plot RMSE versus N
plt.figure(figsize=(12, 8), dpi=100)
plt.plot(range(1, Nmax+1), RMSE, 'x-')
plt.grid()
plt.xlabel('N')
plt.ylabel('RMSE')
plt.xlim([0, 21])
plt.xticks([2, 5, 10, 15, 20])
plt.title("RMSE versus N")
plt.show()

# Plot MAPE versus N. Note for MAPE smaller better. 
plt.figure(figsize=(12, 8), dpi=80)
plt.plot(range(1, Nmax+1), mape, 'x-')
plt.grid()
plt.xlabel('N')
plt.ylabel('MAPE')
plt.xlim([0, 21])
plt.xticks([2, 5, 10, 15, 20])
plt.title("MAPE versus N")
plt.show()

# View Validation data
ax = train.plot(x='date', y='c', style='b-', grid=True)
ax = cv.plot(x='date', y='c', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='c', style='g-', grid=True, ax=ax)
ax = cv.plot(x='date', y='est_N1', style='r-', grid=True, ax=ax)
ax = cv.plot(x='date', y='est_N2', style='m-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test', 'predictions with N=1', 'predictions with N=2'])
ax.set_xlabel("date")
ax.set_ylabel("LKR")
plt.title("Validation Data Results")
plt.show()

# Set optimum N
N_opt = 1

# On test data
est_list = get_preds_mov_avg(df, 'c', N_opt, 0, num_train+num_cv)
test.loc[:, 'est' + '_N' + str(N_opt)] = est_list
print("RMSE = %0.3f" % math.sqrt(mean_squared_error(est_list, test['c'])))
print("MAPE = %0.3f%%" % get_mape(test['c'], est_list))
test.head()

# On test data
N_opt2 = 2
est_list = get_preds_mov_avg(df, 'c', N_opt2, 0, num_train+num_cv)
test.loc[:, 'est' + '_N' + str(N_opt2)] = est_list
print("RMSE = %0.3f" % math.sqrt(mean_squared_error(est_list, test['c'])))
print("MAPE = %0.3f%%" % get_mape(test['c'], est_list))
test.head()

# View Test Data results for all 3 data sets
ax = train.plot(x='date', y='c', style='b-', grid=True)
ax = cv.plot(x='date', y='c', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='c', style='g-', grid=True, ax=ax)
ax = test.plot(x='date', y='est_N{}'.format(N_opt), style='r-', grid=True, ax=ax)
ax = test.plot(x='date', y='est_N{}'.format(N_opt2), style='c-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test', 'predictions with N_opt={}'.format(N_opt), 'predictions with N_opt={}'.format(N_opt2)])
ax.set_xlabel("date")
ax.set_ylabel("LKR")
plt.title("Test Data Results")
plt.show()

# ZOOM to test data set
ax = train.plot(x='date', y='c', style='bx-', grid=True)
ax = cv.plot(x='date', y='c', style='yx-', grid=True, ax=ax)
ax = test.plot(x='date', y='c', style='gx-', grid=True, ax=ax)
ax = test.plot(x='date', y='est_N{}'.format(N_opt), style='rx-', grid=True, ax=ax)
ax = test.plot(x='date', y='est_N{}'.format(N_opt2), style='cx-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test', 'predictions with N_opt={}'.format(N_opt), 'predictions with N_opt={}'.format(N_opt2)])
ax.set_xlabel("date")
ax.set_ylabel("LKR")
ax.set_xlim([date(2020, 8, 1), date(2020, 10, 1)])
#ax.set_ylim([100, 140])
ax.set_title('Zoom in to test set')
plt.show()