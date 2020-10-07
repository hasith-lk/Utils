""" # How to USE
# to run the script, open a new session to atrad using web browser
# Login to atrad using username and password
# using chrome dev tools, find jsession id used to make requests to server
# copy it to JSESSIONID variable.
# default from date is 1st of JULY 2019. use epoch to change it
# default time resolution is dates (D)

import requests
import json
import time
import os
import multiprocessing
from multiprocessing import Pool
from datetime import datetime

def runRequest(stockCode):
    print("Process {} done".format(stockCode))

if __name__ == '__main__':    
    nfilepath = "D:\\Personal\\share\BasicData\\ALL-stocks.txt"
    savefolderPath = "D:\\Personal\\share\\data"

    baseUrl = 'https://online.capitaltrustholding.com/atsweb/Chart'
    JSESSIONID='B803747635DA0262414B260D1A624CC6'

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

    nFile = open(nfilepath, "r")
    Nlines = nFile.readlines()
    Nlines = [line.rstrip() for line in Nlines]
    nFile.close()

    def writeToFile(SLines):
        for sLine in SLines:
            sFileName = savefolderPath + '\\'+ sLine + '.txt'
            print(sFileName)
            #print (getStockData(xLine))
            file = open(sFileName, 'w')
            file.write(getStockData(sLine).lstrip())
            file.close()

    class DownloadStockData(multiprocessing.Process):
        def __init__(self, stockCode):
            super(DownloadStockData, self).__init__()
            self.stockCode = stockCode

        def run(self):
            print("Process {} done".format(self.stockCode))

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

    now = datetime.now()
    print('Start data download at {}:{} {}'.format(now.hour, now.minute, now.second))

    #writeToFile(Nlines)


    agents = 5
    chunksize = 3
    with Pool(processes=agents) as pool:
        result = pool.map(runRequest, Nlines, chunksize)

    exit = True
    while(not exit):
        if counter < len(Nlines):
            p1 = DownloadStockData(counter)
            p1.start()
            counter = counter + 1
        if counter < len(Nlines):
            p2 = DownloadStockData(counter)
            p2.start()
            counter = counter + 1
        if counter < len(Nlines):
            p3 = DownloadStockData(counter)
            p3.start()
            counter = counter + 1
        if counter < len(Nlines):
            p4 = DownloadStockData(counter)
            p4.start()
            counter = counter + 1
        if counter < len(Nlines):
            p5 = DownloadStockData(counter)
            p5.start()
            counter = counter + 1
        if counter == len(Nlines):
            exit = True;


    #print(len(Nlines))
    #for line in Nlines:
    #    sFileName = savefolderPath + '\\'+ line + '.txt'
        #print(sFileName)

    now = datetime.now()
    print('Complete data download at {}:{} {}'.format(now.hour, now.minute, now.second))
    
 """