



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
from sklearn.metrics import mean_squared_error,mean_absolute_error



#loading the data 
cfv=pd.read_csv('cfv.csv')



#time indexing the data for time series analysis
cfv['Date'] = pd.to_datetime(cfv['Month_wise'])
cfv = cfv.set_index('Date')
ts=cfv.Total_cfv
ts.index = ts.index + pd.offsets.MonthEnd(0)


#Stationarity test using Dickey-Fuller Method
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    window=12
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    
    #Perform Dickey-Fuller test (one of the statistical tests for checking stationarity):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(ts)






#ACF/PACF plots
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts, nlags=40)
lag_pacf = pacf(ts, nlags=40, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.stem(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#decomposing the timeseries
result = seasonal_decompose(ts, model='additive')
result.plot()
plt.show()


#checking the stationarity of residuals
residual = result.resid
ts_decompose = residual
ts_decompose.dropna(inplace=True)
test_stationarity(ts_decompose)




#optimizing SARIMA parameters by grid search and selecting for least AIC score
p=q=range(0,2) #from ACF and PACF plots
d=[0] # for non-differenced series
P=Q=range(0,2)
D=range(0,2)


pdq=list(itertools.product(p, d, q))
seasonality=[12]
seasonal_pdq=list(itertools.product(P, D, Q,seasonality))




aic=[]

SARIMA_param=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(ts,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            #print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            SARIMA_param.append([param,param_seasonal])
            aic.append(results.aic)
        
          
        except:
            continue
        
optimum_SARIMA_param_aic=SARIMA_param[(aic.index(min(aic)))]
print('The best parameters for model are: ',(optimum_SARIMA_param_aic,min(aic)))



#fitting the model with best parameters
mod = sm.tsa.statespace.SARIMAX(ts,
                                order=(1,0,1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

mod_fit = mod.fit()

#running diagnostics 
mod_fit.plot_diagnostics(figsize=(8, 8))
plt.show()

#splitting the dataset into training and testing data
size=int(len(ts.index)*0.80)+2
start_index_test=(str(ts.index[size])[:10])

#predicting on testing data
pred = mod_fit.get_prediction(start=pd.to_datetime(start_index_test), dynamic=False)
#pred = mod_fit.get_prediction(start=pd.to_datetime('2017-02-01'), dynamic=False)
pred_ci = pred.conf_int()


#predicting on training data
start_index_train=(str(ts.index[0])[:10])
pred_train = mod_fit.get_prediction(start=pd.to_datetime(start_index_train), end=pd.to_datetime(start_index_test),dynamic=False)
pred_ci_train = pred_train.conf_int()


#plot of actual vs predicted
ax = ts.plot(label='Actual')
pred.predicted_mean.plot(ax=ax, label='One-step ahead prediction', alpha=.7, figsize=(7, 7),color='red')
pred_train.predicted_mean.plot(ax=ax, label='One-step ahead prediction for train', alpha=.7, figsize=(7, 7),color='green')

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)




ax.set_ylabel("Total CFVs", fontname="Times New Roman", fontsize=12)  
ax.set_xlabel("Month", fontname="Times New Roman", fontsize=12)

  
plt.legend()
plt.show()

#SARIMA evalution
y_forecasted = pred.predicted_mean
y_truth = ts[start_index_test:]

y_forecasted_train=pred_train.predicted_mean
y_truth_train = ts[start_index_train:start_index_test]


rmse_test_sarima=np.sqrt(mean_squared_error(y_forecasted, y_truth))
rmse_train_sarima=np.sqrt(mean_squared_error(y_forecasted_train, y_truth_train))

mae_test_sarima= mean_absolute_error(y_forecasted, y_truth)
mae_train_sarima=mean_absolute_error(y_forecasted_train, y_truth_train)


print('RMSE for train: %.3f' %(rmse_train_sarima))
print('RMSE for test: %.3f' %(rmse_test_sarima))



#forecasting on the next 12 months
pred_uc = mod_fit.get_forecast(steps=12)
pred_ci = pred_uc.conf_int(alpha=0.05)
forecast_series=round(pred_uc.predicted_mean)
print(forecast_series)

pred_ci[pred_ci < 0] = 0
pred_ci=pred_ci.round()

ax = ts.plot(label='Actual', figsize=(7, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast',color='red')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)



ax.set_ylabel("Total CFVs", fontname="Times New Roman", fontsize=12)  
ax.set_xlabel("Month", fontname="Times New Roman", fontsize=12)


  
plt.legend()
plt.show()



#Prophet Model
from fbprophet import Prophet
plt.style.use('fivethirtyeight')

#loading the data 
cfv_fb=pd.read_csv('cfv.csv')
cfv_fb['Month_wise'] = pd.DatetimeIndex(cfv_fb['Month_wise'])
cfv_fb = cfv_fb.rename(columns={'Month_wise': 'ds',
                        'Total_cfv': 'y'})

df=cfv_fb
df.ds = df.ds + pd.offsets.MonthEnd(0) 

#Fit the model
model=Prophet(interval_width=0.95)
model.fit(df)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

#plot actual vs predicted
model.plot(forecast[:84],xlabel='Month', ylabel='Total CFVs')
model.plot(forecast,xlabel='Month', ylabel='Total CFVs')
model.plot_components(forecast)

#prediction for next 12 months
#fc_series=round(forecast.yhat[84:])
fc_series=round(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][84:])

#predictions on the original data
pred_fb=forecast.yhat[:84]


#split into train and test set
size = int(len(pred_fb) * 0.80)
X=df.y
train_fb,test_fb=X[0:size],X[size:len(X)]
pred_train_fb, pred_test_fb = pred_fb[0:size], pred_fb[size:len(pred_fb)]


#Evaluate Prophet model 
mae_test_fb=mean_absolute_error(pred_test_fb, test_fb)
mae_train_fb=mean_absolute_error(pred_train_fb, train_fb)

rmse_test_fb=np.sqrt(mean_squared_error(pred_test_fb, test_fb))
rmse_train_fb=np.sqrt(mean_squared_error(pred_train_fb, train_fb))
 
#create a combined performance dataframe for SARIMA and Prophet
error = {'Prophet' : pd.Series([rmse_train_fb,rmse_test_fb,mae_train_fb,mae_test_fb], index =['rmse train', 'rmse test','mae train','mae test']),
          'SARIMA':  pd.Series([rmse_train_sarima,rmse_test_sarima,mae_train_sarima,mae_test_sarima], index =['rmse train', 'rmse test','mae train','mae test']) }
error_df = pd.DataFrame(error) 

