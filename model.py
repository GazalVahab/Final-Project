
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('Smart_Home_Energy_Weather.csv')
time_index = pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')  
time_index = pd.DatetimeIndex(time_index)
data = data.set_index(time_index)
data = data.dropna()
energy_data = data.filter(items=['use [kW]', 'gen [kW]', 'Dishwasher [kW]',
       'Furnace [kW]', 'Home office [kW]', 'Fridge [kW]',
       'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',
       'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',
       'Microwave [kW]', 'Living room [kW]'])

data_per_Hour = energy_data.resample('H').mean()

#  Check Stationarity
#  Augmented Dickey–Fuller test
from statsmodels.tsa.stattools import adfuller
def adf_test(dataset):
  dftest = adfuller(dataset, autolag = 'AIC')
  print(" P-Value : ", dftest[1])
adf_test(data_per_Hour["use [kW]"])

stepwise_fit = auto_arima(data_per_Hour["use [kW]"], trace= True,suppress_warnings=True)           

train=data_per_Hour.iloc[:-30]
test=data_per_Hour.iloc[-30:]


model=ARIMA(data_per_Hour['use [kW]'],order=(1,1,5))
model=model.fit()


index_future_dates=pd.date_range(start='2016-12-18',end='2017-12-18')
#print("index_future_dates----------------->",index_future_dates)
pred=model.predict(start=len(data_per_Hour),end=len(data_per_Hour)+365,typ='levels').rename('ARIMA Predictions')
#print("pred------>",pred)
pred.index=index_future_dates
#print("pred.index",pred.index)


# converting dataframe to a csv file
df = pd.DataFrame(pred, pred.index)
#print(df)
df.to_csv('mout.csv',header = True)