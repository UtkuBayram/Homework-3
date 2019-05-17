import pandas as pd
from pandas import Series
from  matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt
import csv
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pickle


def estimate_holt(df, seriesname, alpha=0.2, slope=0.1, trend="add"):
    numbers = np.asarray(df[seriesname], dtype='float')
    model = Holt(numbers)
    fit = model.fit(alpha, slope, trend)
    estimate = fit.forecast(2)[-1]
    print("Dollar estimation:", estimate)
    return estimate

def decomp(frame,name,f,mod='Additive'):
    #frame['Date'] = pd.to_datetime(frame['Date'])
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def calc_RMSE(tqname,seriesname):
    dfList = df[seriesname].tolist()
    actual = round(dfList[-1],4)
    estimate = round( tqname(df, seriesname, alpha=0.2, slope=0.1, trend="add"), 4)
    RMSE = sqrt(mean_squared_error([actual],[estimate]))
    return RMSE


df= pd.read_csv("HW03_USD_TRY_Trading.txt" , delimiter = '\t')


seriesname = "Close"
series = df[seriesname]
test_stationarity(series)
print(series.tail())


#function for calculating everything needed to answer the questions 1 and 2
def calculate_everything(df):
    size = len(series)
    train = series[-10:size-5]
    trainarray= np.asarray(train)
    test = series[size-5:]
    testarray = np.asarray(test)
    print("Training data:", trainarray, "Test data:", testarray)

    RMSE_Holt = calc_RMSE(estimate_holt,seriesname)

    print("Estimate holt RMSE:",RMSE_Holt)

    dfList = df[seriesname].tolist()
    actual = round(dfList[-1],4)

    dfList2 = df[seriesname].tolist()
    naive = round(dfList2[-1],4)

    estimate =round( estimate_holt(df, seriesname, alpha=0.2, slope=0.1, trend="add"), 4)
    estimate_naive = naive

    print("estimate HOLT:", estimate)
    print("actual:" ,actual)
    print("Naive estimate:",estimate_naive)
    RMSE = sqrt(mean_squared_error([actual],[estimate]))
    RMSE_naive = sqrt(mean_squared_error([actual],[estimate_naive]))
    print("RMSE for holt estimation:",RMSE)
    print("RMSE for naive estimation:",RMSE_naive)

    
    return RMSE,RMSE_naive

print(calculate_everything(df))


#simple exponential smoothing estimation
data = df[seriesname]
model = SimpleExpSmoothing(data)
model_fit = model.fit()
yhat = model_fit.predict(-10, len(data))
print(yhat)


#
#print(df.describe())

##Decomposition of data and observation of trend seasonality and residual
#
from statsmodels.tsa.seasonal import seasonal_decompose
series = df[seriesname]
result = seasonal_decompose(series,freq = 1440, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
result.plot()
plt.show()

#discard rows with no trading data
df = df[~(df == 0).any(axis=1)]
df = df.dropna()
#print(df)

##Question 2

df2 = df.iloc[::60]
df3 = df.iloc[59::60]
df_all_rows = pd.concat([df2, df3])
###discard rows with no trading data
df_all_rows = df_all_rows[~(df == 0).any(axis=1)]
df_all_rows = df_all_rows.dropna()
print(df_all_rows)

seriesname = "Close"
series = df_all_rows[seriesname]
test_stationarity(series)

print(calculate_everything(df_all_rows))

#3A#smoothing by A simple (one-sided) moving average
df = df.set_index(["Day","Time"])
smoothed_data = df.rolling(window=60).mean()
smoothed_data = smoothed_data.dropna()
#print(smoothed_data.head(61))
calculate_everything(smoothed_data)
df = df.reset_index(["Day","Time"])

seriesname = "Close"
series = df[seriesname]
test_stationarity(series)
print(series.tail())

###3C One random (representative) pick from each hour
df2 = (df.sample(n= 10))
calculate_everything(df2)
seriesname = "Close"
series = df2[seriesname]
test_stationarity(series)
print(series.tail())


