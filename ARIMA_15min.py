import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
import csv
import sys
from statsmodels.tsa.arima_model import ARMA

def loadDataSet():
    file2 = file("all_surround_cells.csv","rb")
    reader2 = csv.reader(file2)
    surround_cells = []
    for line in reader2:
        surround_cells.append(str(round(float(line[1]),5))+'_'+str(round(float(line[0]),5)))

    traffic_dataFrame = pd.read_csv("/home/johnson/tensorflow/row Data/nj06downbsloc15min_new.csv")

    MyTrafficFrame = traffic_dataFrame.reindex(columns=surround_cells)        #[2861 rows * 39 columns]

    ##miss data operation
    MyTrafficFrame =  MyTrafficFrame.dropna(axis = 1,thresh = 935)           #[2861 rows * 19 columns]
    MyTrafficFrame = MyTrafficFrame.interpolate()                             #interpolate values
                                     
    return MyTrafficFrame,MyTrafficFrame['32.05278_118.77965']                #type(MyTrafficFrame.values) == <type 'numpy.ndarray'>  

def normalization(x):                                                         #type of input is <type 'numpy.ndarray'>
    min_max_scaler = preprocessing.MinMaxScaler()   
    return min_max_scaler.fit_transform(x)

def autocorrelation(x,lags):                                                  #Temporal correlation  
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:] - x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
            /(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
    return result


def testStationarity(ts):
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

def draw_acf_pacf(ts):
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(211)
    plot_acf(ts ,ax=ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(ts,ax=ax2)
    plt.show()


def proper_model(data_ts, maxLag):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel


_,traffic = loadDataSet() 
rng = pd.date_range('6/1/2014 00:00',periods = 2861,freq = '15Min')
traffic = normalization(traffic)
traffic_train = traffic[:2000]
traffic_test = traffic[2000:]                                                #type(traffic) = <class 'pandas.core.series.Series'>   
traffic_train = pd.Series(traffic_train,index = rng[:2000])
traffic_test = pd.Series(traffic_test,index = rng[2000:])
traffic = pd.Series(traffic)
traffic_diff1 = traffic_train.diff(1)

#print proper_model(traffic_train,20)

#print testStationarity(traffic)
#draw_acf_pacf(traffic)

model = ARMA(traffic_train, order=(3, 5)) 
result_arma = model.fit( disp=-1, method='css')
predict_ts_train = result_arma.predict()
predict_ts_test = result_arma.predict('6/21/2014 20:00:00','6/30/2014 19:00:00',dynamic = True)

fig1 = plt.figure(1)
plt.plot(traffic_train,'r')
plt.plot(predict_ts_train,'b--')
fig2 = plt.figure(2)
plt.plot(traffic_test,'r')
plt.plot(predict_ts_test,'b--')
plt.show()

traffic_test.to_csv('/home/johnson/tensorflow/pic/15min/result_15min/ARIMA_test_result')
predict_ts_test.to_csv('/home/johnson/tensorflow/pic/15min/result_15min/ARIMA_predict_result')
print metrics.mean_squared_error(traffic_test,predict_ts_test)              ##MSE = 0.018879
print metrics.mean_absolute_error(traffic_test,predict_ts_test)             ##MAE = 0.101240
