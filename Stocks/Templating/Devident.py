import pandas as pd
import numpy as np

stockName = 'JKH.N0000'

stockListPath = 'D:\\Personal\\share\\BasicData\\N-stocks.txt'
execelFile = 'D:\\Personal\\share\\Financial\\CSE_Dividends Updated_17-03-2020.xlsx'
dataFolderPath = 'D:\\Personal\\share\\data'

df = pd.read_excel(execelFile)

df = df[pd.notna(df['Code'])]
df = df['Code'].trim()

#df.set_index(['Code'])
df.head()


dataFile = open(stockListPath, "r")
dataLines = dataFile.readlines()
dataLines = [line.rstrip() for line in dataLines]
dataFile.close()

detail = []
for stockName in dataLines:
    dataFileName = dataFolderPath + '\\'+ stockName + '.txt'
    dfc = pd.read_json(dataFileName)
    dfc['t'] = pd.to_datetime(dfc['t'], unit='s')
    dfc['t'] = dfc['t'].dt.date

    index = dfc.last_valid_index()

    if len(dfc.index) > 0:
        detail.append([stockName, dfc.iloc[index]['c']])
    
columns = ['Stock', 'Close']
data = pd.DataFrame(detail, columns=columns)
#data.set_index(['Stock'])

#print(data.index)
#data.head()
#data.dtypes

datar = df[['Code', 2019]]

data = data.sort_values(by=['Stock'])
datar = datar.sort_values(by=['Code'])

data.head()
datar.head()


result = pd.merge(data, datar, left_on=['Stock'], right_on=['Code'], how='inner')
result['yeild'] = result[2019] / result['Close']
result.head(20)
resultFolderPath = 'D:\\Personal\\share\\results'

result.to_csv(resultFolderPath + '\\Dev_Yeild.csv')

print('done')