import requests
import json
import pandas as pd

nfilepath = "D:\\Personal\\share\BasicData\\ALL-stocks.txt"
savefolderPath = "D:\\Personal\\share\\data_TMP"
baseUrl = "https://www.cse.lk/api/companyInfoSummery"

def getStockData(stockId):
    payload = {'symbol':stockId}
    headers = {'Cookie': '_ga=GA1.2.1335521007.1590466015; AWSALB=V8/9xGozOwTuqpVwqYccS2+YC1FGuYkwEmYXfwJIZZqJhH60+ixNzP9UXBOER9ibW2G7SHz03sEVjGbtz4wvk66cG93n7sQRL1g2RXUPfbzcQxksjq/EtF87SOtH; AWSALBCORS=V8/9xGozOwTuqpVwqYccS2+YC1FGuYkwEmYXfwJIZZqJhH60+ixNzP9UXBOER9ibW2G7SHz03sEVjGbtz4wvk66cG93n7sQRL1g2RXUPfbzcQxksjq/EtF87SOtH'}
    response = requests.post(baseUrl, params=payload, headers=headers)
    if response.status_code == 200 :
        return response.text
    else:
        return ''

nFile = open(nfilepath, "r")
Nlines = nFile.readlines()
Nlines = [line.rstrip() for line in Nlines]
nFile.close()

def writeToFile(SLines):
    sFileName = savefolderPath + '\\CompanyBetaInfo.csv'
    companyData = []
    print(sFileName)
    for sLine in SLines:
        print('Loading {}'.format(sLine))
        companyRawData = getStockData(sLine).lstrip()
        companyRawData = json.loads(companyRawData)     #Convert to readable format
        betaRecord = companyRawData['reqSymbolBetaInfo']
        betaRecord['securityCode'] = sLine
        companyData.append(betaRecord)
        #print(companyData)
    df = pd.DataFrame.from_dict(companyData)
    df.set_index('securityCode', inplace=True)
    df.to_csv(sFileName)
    return df

print('Starts')
writeToFile(Nlines)
print('End')


'''
import pandas as pd

df = pd.read_csv("D:\\Personal\\share\\data_TMP\\CompanyBetaInfo.txt",)
df.head()
'''
