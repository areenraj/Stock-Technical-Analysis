import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
from datetime import datetime
import yfinance as yf

yf.pdr_override()

class Stock:
    def __init__(self, name, ticker, start_date, end_date):
        self.name = name
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.returns = pdr.get_data_yahoo(ticker, start_date - dt.timedelta(days=30), end_date + dt.timedelta(days=30))
        
        
def calcrs(stock):
    delta = stock.returns['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0)

    emaup = gain.ewm(span=14,adjust=False).mean()
    emadown = abs(loss.ewm(span=14, adjust=False).mean())

    rsi = emaup/emadown
    
    rsi = 100 - (100/(1+rsi))
    return rsi

def calcmacd(stock):
    macd = stock.returns['Close'].ewm(span=12).mean() - stock.returns['Close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    return macd, signal


def calcadx(stock):

    plusdi = stock.returns['High'].diff(1)
    minusdi = stock.returns['Low'].diff(-1)

    plusdi = plusdi.clip(lower=0.0)
    minusdi = minusdi.clip(lower=0.0)

    for i in range(len(plusdi)):
        if plusdi.iloc[i]>minusdi.iloc[i]:
            minusdi.iloc[i] = 0.0
        elif plusdi.iloc[i] < minusdi.iloc[i]:
            plusdi.iloc[i] = 0.0

    df = stock.returns[['Close', 'High', 'Low']].copy()

    df['TR_TMP1'] = df['High'] - df['Low']
    df['TR_TMP2'] = np.abs(df['High'] - df['Close'].shift(1))
    df['TR_TMP3'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)

    TR = df['TR']

    atr = TR
    plusdm = plusdi
    minusdm = minusdi

    atr1 = 0.0
    plusdm1 = 0.0
    minusdm1 = 0.0

    for i in range(14):
        atr1  += TR.iloc[i+1]
        plusdm1 += plusdi.iloc[i+1]
        minusdm1 += minusdi.iloc[i+1]

    atr.iloc[14] = atr1
    minusdm.iloc[14] = minusdm1
    plusdm.iloc[14] = plusdm1

    for i in range(len(plusdi)-15):
        atr.iloc[i+15] = (atr.iloc[i+14]*13 + TR.iloc[i+15])/14
        minusdm.iloc[i+15] = (minusdm.iloc[i+14]*13 + minusdi.iloc[i+15])/14
        plusdm.iloc[i+15] = (plusdm.iloc[i+14]*13 + plusdi.iloc[i+15])/14

    plusdm = (plusdm/atr)*100
    minusdm = (minusdm/atr)*100

    dx = (abs(plusdm-minusdm)/abs(plusdm+minusdm))*100
    
    adx = dx

    adx1 = 0.0
    
    for i in range(14):
        adx1 += dx.iloc[i+14]

    adx[27] = adx1/14.0

    for i in range(len(plusdi)-28):
        adx.iloc[i+28] = (adx.iloc[i+27]*13 + dx.iloc[i+28])/14

    return adx

def calcadl(stock):

    high = stock.returns['High']
    low = stock.returns['Low']
    close = stock.returns['Close']
    volume = stock.returns['Volume']

    mfm = ((close - low) - (high - close)/(high - low))

    mfv = mfm*volume

    print(mfv)

    adl = mfv.cumsum()

    return adl

def calcstochosc(stock):
    high = stock.returns['High']
    low = stock.returns['Low']
    close = stock.returns['Close']

    minlow = low.rolling(window=14).min()
    maxhigh = high.rolling(window=14).max()

    perk = ((close - minlow)/(maxhigh-minlow))*100

    fast_perk = perk.rolling(window=3).mean()

    return perk,fast_perk

stock1 = Stock("Nestle", "NESTLEIND.NS", dt.date(2022,1,1), dt.date(2022,12,31))
stock2 = Stock("Reliance", "RELIANCE.NS", dt.date(2022,1,1), dt.date(2022,12,31))

stocks = [stock1,stock2]

for i in range(len(stocks)):
    rsi = calcrs(stocks[i])
    macd, signal = calcmacd(stocks[i])
    adx = calcadx(stocks[i])
    perk, fast_perk = calcstochosc(stocks[i])

    fig1 = plt.figure(figsize=(20,10))
    fig1.suptitle(stocks[i].name, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    price = plt.subplot(311)
    price.plot(stocks[i].returns['Close'][(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#0A4D68')
    price.set_title("Close Price")
    price.set_xlabel("Date")
    price.set_ylabel("Price")
    price.figure.set_facecolor('#FFE7CC')
    price.set_facecolor('#B6EADA')

    relst = plt.subplot(312)
    relst.axhline(y=70, color="black", linestyle="--")
    relst.axhline(y=30, color="black", linestyle="--")
    relst.plot(rsi[(stocks[i].start + dt.timedelta(days=5)).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#0A4D68')
    relst.set_title("Relative Strength Index")
    relst.set_xlabel("Date")
    relst.set_ylabel("RSI")
    relst.set_facecolor('#B6EADA')

    macovdiv = plt.subplot(313)
    macovdiv.set_title("Moving Average Convergence And Divergence")
    macovdiv.axhline(y=0, color="black", linestyle="--")
    macovdiv.plot(macd[(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#0A4D68')
    macovdiv.plot(signal[(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#FFC93C')
    macovdiv.set_xlabel("Date")
    macovdiv.set_ylabel("MACD")
    macovdiv.set_facecolor('#B6EADA')

    plt.show()

    fig2 = plt.figure(figsize=(20,10))
    fig2.suptitle(stocks[i].name, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    price = plt.subplot(311)
    price.plot(stocks[i].returns['Close'][(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#0A4D68')
    price.set_title("Close Price")
    price.set_xlabel("Date")
    price.set_ylabel("Price")
    price.figure.set_facecolor('#FFE7CC')
    price.set_facecolor('#B6EADA')

    adirx = plt.subplot(312)
    adirx.plot(adx[(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#0A4D68')
    adirx.set_title("Average Directional Index")
    adirx.set_xlabel("Date")
    adirx.set_ylabel("ADX")
    adirx.figure.set_facecolor('#FFE7CC')
    adirx.set_facecolor('#B6EADA')

    stococ = plt.subplot(313)
    stococ.axhline(y=80, color="black", linestyle="--")
    stococ.axhline(y=20, color="black", linestyle="--")
    stococ.plot(perk[(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#0A4D68')
    stococ.plot(fast_perk[(stocks[i].start).strftime("%Y-%m-%d"):stocks[i].end.strftime("%Y-%m-%d")], color='#FFC93C')
    stococ.set_title("Stochastic Oscillator")
    stococ.set_xlabel("Date")
    stococ.set_ylabel("Stoch Osc")
    stococ.figure.set_facecolor('#FFE7CC')
    stococ.set_facecolor('#B6EADA')

    plt.show()



