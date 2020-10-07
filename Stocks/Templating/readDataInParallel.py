# How to USE
# to run the script, open a new session to atrad using web browser
# Login to atrad using username and password
# using chrome dev tools, find jsession id used to make requests to server
# copy it to JSESSIONID variable.
# default from date is 1st of JULY 2019. use epoch to change it
# default time resolution is dates (D)

import requests
import time
import os
from datetime import datetime
from multiprocessing import Pool

nfilepath = "D:\\Personal\\share\BasicData\\ALL-stocks.txt"
savefolderPath = "D:\\Personal\\share\\data"

baseUrl = 'https://online.capitaltrustholding.com/atsweb/Chart'
JSESSIONID='0A16C799C3527B41334EE97257F523D3'

#fromDate = '1561986000'
fromDate = '1572570001'  # 2019-9-1
currentTimeepoch = int(time.time())

def getStockData(stockId):
    payload = {'action':'history', 'format':'json', 'symbol':stockId, 'resolution':'D', 'from':fromDate, 'to': currentTimeepoch}
    headers = {'Cookie': 'theme=black; JSESSIONID='+JSESSIONID+'; username=; role=OnlineUser; broker_code=DSA; watchID=1219; _ga=GA1.2.2028623637.1590156677; _gid=GA1.2.1252743755.1595909834; i18next=en'}
    response = requests.get(baseUrl, params=payload, headers=headers)
    if response.status_code == 200 :
        return response.text
    else:
        return ''

def writeToFile(stockId):
    sFileName = savefolderPath + '\\'+ stockId + '.txt'
    print(sFileName)
    #print (getStockData(xLine))
    file = open(sFileName, 'w')
    file.write(getStockData(stockId).lstrip())
    file.close()
    return "{} Done".format(stockId)


if __name__ == '__main__':

    print('Start')
    print('Backup Old Data')
    backupStatus = False
    if  os.listdir(savefolderPath):
        dateTimeObj = datetime.now()
        backupFolderName=savefolderPath + "_{}_{}_{}_{}_{}_{}".format(dateTimeObj.year,dateTimeObj.month,dateTimeObj.day,
        dateTimeObj.hour,dateTimeObj.minute,dateTimeObj.second);
        os.rename(savefolderPath,backupFolderName)
        os.makedirs(savefolderPath)
        print('Backup Done')
        backupStatus = True
    else:
        print('No Backup')
        backupStatus = True    

    # Read share codes from files
    nFile = open(nfilepath, "r")
    Nlines = nFile.readlines()
    Nlines = [line.rstrip() for line in Nlines]
    nFile.close()

    # Define the dataset
    dataset = Nlines
    
    now = datetime.now()
    print('Start data download at {}:{} {}'.format(now.hour, now.minute, now.second))

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    agents = 5
    chunksize = 3
    with Pool(processes=agents) as pool:
        result = pool.map(writeToFile, dataset, chunksize)

    now = datetime.now()
    print('Complete data download at {}:{} {}'.format(now.hour, now.minute, now.second))
