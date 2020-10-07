import datetime
from Templating.StocksCommon import CalcBuyCost, CalcSellReturn, BuyQtyForPrice

START_TRADING_ON = datetime.date(2020, 5, 14)
NO_OF_SHARES_KEY = 'NoOfShares'
CASH_KEY = 'Cash'
CASH_VALUE = 100000

def Exec_Buy(price, stockAccount):
    nShares, costDiff  = BuyQtyForPrice(price, stockAccount[CASH_KEY])
    stockAccount[CASH_KEY] = costDiff
    stockAccount[NO_OF_SHARES_KEY] = nShares    
    return stockAccount

def Exec_Sell(price, stockAccount):
    amount = CalcSellReturn(stockAccount[NO_OF_SHARES_KEY], price)
    stockAccount[CASH_KEY] = stockAccount[CASH_KEY] + amount
    stockAccount[NO_OF_SHARES_KEY] = 0
    return stockAccount
