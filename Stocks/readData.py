# How to USE
# to run the script, open a new session to atrad using web browser
# Login to atrad using username and password
# using chrome dev tools, find jsession id used to make requests to server
# copy it to JSESSIONID variable.
# default from date is 1st of JULY 2019. use epoch to change it
# default time resolution is dates (D)

import requests
import json
import time

nfilepath = "D:\\share\\N-stocks.txt"
xfilepath = "D:\\share\\X-stocks.txt"
savefolderPath = "D:\\share\\data"

baseUrl = 'https://online.capitaltrustholding.com/atsweb/Chart'
JSESSIONID='596EBA2F96ADFA366D4F8289A2624593'

fromDate = '1561939200'
currentTimeepoch = int(time.time())

def getStockData(stockId):
    payload = {'action':'history', 'format':'json', 'symbol':stockId, 'resolution':'D', 'from':fromDate, 'to': currentTimeepoch}
    headers = {'Cookie': 'theme=black; JSESSIONID='+JSESSIONID+'; username=; role=OnlineUser; broker_code=DSA; watchID=1219; _ga=GA1.2.2028623637.1590156677; _gid=GA1.2.1252743755.1595909834; i18next=en'}
    response = requests.get(baseUrl, params=payload, headers=headers)
    if response.status_code == 200 :
        return response.text
    else:
        return '' 

nFile = open(nfilepath, "r")
Nlines = nFile.readlines()
Nlines = [line.rstrip() for line in Nlines]
nFile.close()

xFile = open(xfilepath, "r")
Xlines = xFile.readlines()
Xlines = [line.rstrip() for line in Xlines]
xFile.close()

def writeToFile(SLines):
    for sLine in SLines:
        sFileName = savefolderPath + '\\'+ sLine + '.txt'
        print(sFileName)
        #print (getStockData(xLine))
        file = open(sFileName, 'w')
        file.write(getStockData(sLine).lstrip())
        file.close()

writeToFile(Xlines)
writeToFile(Nlines)
print('done')
  
