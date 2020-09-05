import requests
import json
from getpass import getpass
import time

baseUrl = 'https://online.capitaltrustholding.com/atsweb/Chart'
currentTimeepoch = int(time.time())

print(currentTimeepoch)


payload = {'action':'history', 'format':'json', 'symbol':'VONE.N0000', 'resolution':'D', 'from':'1561939200', 'to': currentTimeepoch}
headers = {'Cookie': 'theme=black; JSESSIONID=596EBA2F96ADFA366D4F8289A2624593; username=; role=OnlineUser; broker_code=DSA; watchID=1219; _ga=GA1.2.2028623637.1590156677; _gid=GA1.2.1252743755.1595909834; i18next=en'}

response = requests.get(baseUrl, params=payload, headers=headers)
print(response.url)
print(response.status_code)
print(response.text)