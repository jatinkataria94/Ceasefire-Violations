
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from fbprophet import Prophet
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error,mean_absolute_error


#loading the data 
cfv=pd.read_csv('cfv.csv')
cfv_sectors=pd.read_csv('cfv_sectors_var.csv')
cfv_sectors=cfv_sectors.fillna(0)
cfv_sectors['Month_wise'] = pd.to_datetime(cfv_sectors['Month_wise'])
cfv_sectors.Month_wise = cfv_sectors.Month_wise + pd.offsets.MonthEnd(0) 

#converting data into the required form for Prophet modelling
cfv_sectors.drop(['Not_Mentioned','Total_cfv'],axis=1,inplace=True)

series=[]
for i in range(1,len(cfv_sectors.columns)):
    series.append(cfv_sectors[['Month_wise',cfv_sectors.columns.tolist()[i]]])

sector_name=cfv_sectors.columns.tolist()[1:]
for j in range(len(series)):
    series[j]=series[j].rename(columns={'Month_wise':'ds',sector_name[j]:'y'})


cfv['Month_wise'] = pd.to_datetime(cfv['Month_wise'])
cfv = cfv.rename(columns={'Month_wise': 'ds',
                        'Total_cfv': 'y'})

df=cfv

#define a model fitting function
def run_prophet(timeserie,sect):
    model = Prophet(interval_width=0.95)
    model.fit(timeserie)
    forecast = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(forecast)
    fig=model.plot(forecast,xlabel='Month', ylabel='Total CFVs')
    ax=fig.gca()
    ax.set_title(sect)
    model.plot_components(forecast)
    return forecast



fc_sector_series=[]
error_sector=[]

#define a function to fit and evaluate multiple time series
def sector_wise_prophet(series):
    for i in range(len(series)):
        fc_sector = run_prophet(series[i],sector_name[i])
        fc_sector_series.append(abs(round(fc_sector.yhat[84:])))
        
        pred=fc_sector.yhat[:84]
        pred[pred < 0] = 0
        pred=pred.round()
        
        #split into train and test set
        size = int(len(pred) * 0.80)
        X=series[i].y
        train,test=X[0:size],X[size:len(X)]
        pred_train, pred_test = pred[0:size], pred[size:len(pred)]
        
        
        #Evaluate model by rmse
        rmse_test=np.sqrt(mean_squared_error(pred_test, test))
        rmse_train=np.sqrt(mean_squared_error(pred_train, train))
        
        mae_test=mean_absolute_error(pred_test, test)
        mae_train=mean_absolute_error(pred_train, train)
        
        error_sector.append(pd.DataFrame([rmse_train,rmse_test,mae_train,mae_test],index=['rmse train', 'rmse test','mae train','mae test'],columns=[sector_name[i]]))
        
    return fc_sector_series,error_sector
    

#create dataframes to store forecast and performance values        
forecast_sectors_cfv,error=sector_wise_prophet(series)

forecast_sectors_cfv=pd.concat(forecast_sectors_cfv, axis=1)

forecast_sectors_cfv.columns = sector_name
forecast_sectors_cfv['Total_cfv'] = forecast_sectors_cfv[list(forecast_sectors_cfv.columns)].sum(axis=1)


error_sector_df=pd.concat(error, axis=1)
