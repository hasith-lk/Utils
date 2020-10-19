import math
import numpy as np
import pandas as pd
import seaborn as sns
import time

from matplotlib import pyplot as plt
#from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from xgboost import XGBRegressor

stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

shareCode='COMB.N0000'

#### Input params ##################
test_size = 0.2                # proportion of dataset to be used as test set
cv_size = 0.2                  # proportion of dataset to be used as cross-validation set
N = 7                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

n_estimators = 100             # for the initial model before tuning. default = 100
max_depth = 3                  # for the initial model before tuning. default = 3
learning_rate = 0.1            # for the initial model before tuning. default = 0.1
min_child_weight = 1           # for the initial model before tuning. default = 1
subsample = 1                  # for the initial model before tuning. default = 1
colsample_bytree = 1           # for the initial model before tuning. default = 1
colsample_bylevel = 1          # for the initial model before tuning. default = 1
train_test_split_seed = 111    # 111
model_seed = 100
####################################

df = pd.read_json(dataFolderPath+'\\' + shareCode + '.txt')
df['t'] = pd.to_datetime(df['t'], unit='s')
df['t'] = df['t'].dt.date

# Convert Date column to datetime
df.loc[:, 't'] = pd.to_datetime(df['t'],format='%Y-%m-%d')

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Get month of each sample
df['month'] = df['t'].dt.month

# Sort by datetime
df.sort_values(by='t', inplace=True, ascending=True)

df.tail(10)

ax = df.plot(x='t', y='c', style='b-', grid=True)
ax.set_xlabel("t")
ax.set_ylabel("LKR")
plt.show()

# Get sizes of each of the datasets
num_cv = int(cv_size*len(df))
num_test = int(test_size*len(df))
num_train = len(df) - num_cv - num_test
print("num_train = " + str(num_train))
print("num_cv = " + str(num_cv))
print("num_test = " + str(num_test))

# Split into train, cv, and test
train = df[:num_train]
cv = df[num_train:num_train+num_cv]
train_cv = df[:num_train+num_cv]
test = df[num_train+num_cv:]

print("train.shape = " + str(train.shape))
print("cv.shape = " + str(cv.shape))
print("train_cv.shape = " + str(train_cv.shape))
print("test.shape = " + str(test.shape))

# Converting dataset into x_train and y_train
# Here we only scale the train dataset, and not the entire dataset to prevent information leak
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[['o', 'h', 'l', 'c', 'v']])
print("scaler.mean_ = " + str(scaler.mean_))
print("scaler.var_ = " + str(scaler.var_))
print("train_scaled.shape = " + str(train_scaled.shape))

# Convert the numpy array back into pandas dataframe
train_scaled = pd.DataFrame(train_scaled, columns=['o', 'h', 'l', 'c', 'v'])
train_scaled[['t', 'month']] = train[['t', 'month']]
print("train_scaled.shape = " + str(train_scaled.shape))
train_scaled.head()

# Do scaling for dev set
cv_scaled  = scaler.transform(cv[['o', 'h', 'l', 'c', 'v']])

# Convert the numpy array back into pandas dataframe
cv_scaled = pd.DataFrame(cv_scaled, columns=['o', 'h', 'l', 'c', 'v'])
cv_scaled[['t', 'month']] = cv.reset_index()[['t', 'month']]
print("cv_scaled.shape = " + str(cv_scaled.shape))
cv_scaled.head()

# Do scaling for test set
test_scaled  = scaler.transform(test[['o', 'h', 'l', 'c', 'v']])

# Convert the numpy array back into pandas dataframe
test_scaled = pd.DataFrame(test_scaled, columns=['o', 'h', 'l', 'c', 'v'])
test_scaled[['t', 'month']] = test.reset_index()[['t', 'month']]
print("test_scaled.shape = " + str(test_scaled.shape))
test_scaled.head()


# Combine back train_scaled, cv_scaled, test_scaled together
df_scaled = pd.concat([train_scaled, cv_scaled, test_scaled], axis=0)
df_scaled.head()

# Get difference between high and low of each day
df_scaled['range_hl'] = df_scaled['h'] - df_scaled['l']
df_scaled.drop(['h', 'l'], axis=1, inplace=True)

# Get difference between open and close of each day
df_scaled['range_oc'] = df_scaled['o'] - df_scaled['c']
df_scaled.drop(['o', 'c'], axis=1, inplace=True)

df_scaled.head()

# Add a column 'order_day' to indicate the order of the rows by date
df_scaled['order_day'] = [x for x in list(range(len(df_scaled)))]

# merging_keys
merging_keys = ['order_day']

# List of columns that we will use to create lags
lag_cols = ['c', 'range_hl', 'range_oc', 'v']
lag_cols

""" shift_range = [x+1 for x in range(N)]

for shift in (shift_range):
    train_shift = df_scaled[merging_keys + lag_cols].copy()
    
    # E.g. order_day of 0 becomes 1, for shift = 1.
    # So when this is merged with order_day of 1 in df_scaled, this will represent lag of 1.
    train_shift['order_day'] = train_shift['order_day'] + shift
    
    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
    train_shift = train_shift.rename(columns=foo)

    df_scaled = pd.merge(df_scaled, train_shift, on=merging_keys, how='left') #.fillna(0)
    
del train_shift

# Remove the first N rows which contain NaNs
df_scaled = df_scaled[N:]
    
df_scaled.head() """

