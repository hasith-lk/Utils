import math

def CalcBuyCost(nShares,price):
        SEC = 0.072
        CSE = 0.084
        CDS = 0.024
        STL = 0.3
        BKR = 0.64
        cost = (nShares * price)*((SEC+CSE+STL+BKR)/100)
        if ((nShares * price)*(CDS/100)) > 5 :
            cost = cost + ((nShares * price)*(CDS/100))
        else:
            cost = cost + 5

        return (nShares * price) + cost

def CalcSellReturn(nShares,price):
    SEC = 0.072
    CSE = 0.084
    CDS = 0.024
    STL = 0.3
    BKR = 0.64
    cost = (nShares * price)*((SEC+CSE+STL+BKR)/100)
    if ((nShares * price)*(CDS/100)) > 5 :
        cost = cost + ((nShares * price)*(CDS/100))
    else:
        cost = cost + 5

    return (nShares * price)-cost

def BuyQtyForPrice(price, capital):
    SEC = 0.072
    CSE = 0.084
    CDS = 0.024
    STL = 0.3
    BKR = 0.64
    nShares1 = (capital*100)/((100 + (SEC+CSE+STL+BKR+CDS))*price)
    nShares2 = ((capital-5)*100)/((100 + (SEC+CSE+STL+BKR))*price)

    nShares1 = math.floor(nShares1)
    nShares2 = math.floor(nShares2)

    cost1 = CalcBuyCost(nShares1, price)
    cost2 = CalcBuyCost(nShares2, price)

    costDiff1 = capital - cost1
    costDiff2 = capital - cost2

    if costDiff1 >= 0 and costDiff2 >= 0:
        if costDiff1 > costDiff2:
            return nShares2, costDiff2
        else:
            return nShares1, costDiff1

    if costDiff1 >= 0 and costDiff2 < 0:
        return nShares1, costDiff1
    else:
        return nShares2, costDiff2    
