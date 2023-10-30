import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.dummy import DummyRegressor
#from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from pandas import to_datetime
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
os.getcwd()

df1 = pd.read_csv('crypto_upto_date_prices/combined_file.csv')



def cross_val_folds(data= df1,train_size = 365,test_size=7,coin='BITCOIN'):
    new_df = df1[df1.coin==coin]
    print('shape of new dataframe is',new_df.shape)
    tscv = TimeSeriesSplit(max_train_size = train_size,test_size=test_size)
    cross_val_splits = tscv.split(new_df)
    
    #n_splits = cross_val_splits.get_n_splits
    
    for i, (train_index, test_index) in enumerate(cross_val_splits):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        train_index = train_index
        test_index = test_index
    return train_index,test_index
    

#train_index,text_index = cross_val_folds()
# create test dataset, remove last 7 days of data
def model_prophet(df=df1,train_size=365,test_size=7,coin='BITCOIN'):
  #display(new_df)
    new_df = df[df.coin==coin]
    train_index,test_index = cross_val_folds(train_size=train_size,test_size=test_size)
    # for train_index,test_index:
    # print(f'creating predictions for {coin} for {train_size} training days and {test_size} testing days')
    new_df['date']= to_datetime(df['date'])
    new_df.rename(columns = {'date':'ds','price_in_usd':'y'},inplace=True)
    train_df = new_df.iloc[train_index,:]
    test_df = new_df.iloc[test_index,:]
    # print('shape of training using prophet:',train_df.shape)
    # print('shape of testing using prophet:',test_df.shape)
    model = Prophet()
    model.fit(train_df)

    # define the period for which we want a prediction
    future = list()
    date = (train_df['ds'].iloc[-1])
    # print(date)
    for i in range(test_size):

        date += datetime.timedelta(days=1)
        future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['date']= to_datetime(future['ds'])

    # use the model to make a forecast
    forecast = model.predict(future)
    # summarize the forecast
    #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    # print(f'forecasting for {test_size}days')
    # print(forecast[['ds', 'yhat']])
    # calculate MAE between expected and predicted values
    y_true = new_df[['ds','y']].iloc[test_index]
    # print(y_true)
    y_pred = forecast['yhat']
    mae = mean_absolute_error(y_true['y'], y_pred)
    rmse = mean_squared_error(y_true['y'], y_pred,squared=False)
    #r2_score = r2_score(y_true['y'], y_pred)
    # print('MAE: %.3f' % mae)
    # print('RMSE: %.3f' % rmse)
    rslts = {'model':'Prophet','coin':coin,'train_size':train_size,'test_size':test_size,'mae':mae,'rmse':rmse}
    rslts_final = pd.DataFrame([rslts])
    return y_true,rslts_final
   
    
def dummy_regressor(df=df1,train_size=365,test_size=7,coin='BITCOIN'):
    model = DummyRegressor(strategy='mean')
    print(f'creating predictions for {coin} for {train_size} training days and {test_size} testing days')
    #display(new_df)
    new_df = df[df.coin==coin]
    train_index,test_index = cross_val_folds(train_size=train_size,test_size=test_size)
    new_df['date']= to_datetime(df['date'])
    new_df.rename(columns = {'date':'ds','price_in_usd':'y'},inplace=True)
    train_df = new_df.iloc[train_index,:]
    test_df = new_df.iloc[test_index,:]
    # print('shape of training using dummy regressor:',train_df.shape)
    # print('shape of testing using dummy regressor:',test_df.shape)
    model = DummyRegressor(strategy='mean')
    X = sm.add_constant(train_df['y'])
    #print(X)
    model = sm.OLS(train_df['y'],X)
    results = model.fit()
    preds = (results.params[0] - results.params[1]*test_df['y'])*-1
    # print(preds)
    # calculate MAE between expected and predicted values
    y_true = new_df[['ds','y']].iloc[test_index]
    print(y_true)
    # y_pred = forecast['yhat']
    mae = mean_absolute_error(y_true['y'], preds)
    rmse = mean_squared_error(y_true['y'], preds,squared=False)
    # #r2_score = r2_score(y_true['y'], y_pred)
    # print('MAE: %.3f' % mae)
    # print('RMSE: %.3f' % rmse)
    rslts = {'model':'dummy_regressor','coin':coin,'train_size':train_size,'test_size':test_size,'mae':mae,'rmse':rmse}
    rslts_final = pd.DataFrame([rslts])
    return y_true,rslts_final
    # return mae,rmse
    
def linear_regressor(df=df1,train_size=365,test_size=7,coin='BITCOIN'):
    #model = LinearRegression()
    print(f'creating predictions for {coin} for {train_size} training days and {test_size} testing days')
    #display(new_df)
    new_df = df[df.coin==coin]
    train_index,test_index = cross_val_folds(train_size=train_size,test_size=test_size)
    new_df['date']= to_datetime(df['date'])
    new_df.rename(columns = {'date':'ds','price_in_usd':'y'},inplace=True)
    train_df = new_df.iloc[train_index,:]
    test_df = new_df.iloc[test_index,:]

    model = LinearRegression()
    X = sm.add_constant(train_df['y'])
    #print(X)
    model = sm.OLS(train_df['y'],X)
    results = model.fit()
    preds = (results.params[0] - results.params[1]*test_df['y'])*-1
    #print(preds)
    # calculate MAE between expected and predicted values
    y_true = new_df[['ds','y']].iloc[test_index]
    print(y_true)
    # y_pred = forecast['yhat']
    mae = mean_absolute_error(y_true['y'], preds)
    rmse = mean_squared_error(y_true['y'], preds,squared=False)
    # #r2_score = r2_score(y_true['y'], y_pred)
    #print('MAE: %.3f' % mae)
    #print('RMSE: %.3f' % rmse)
    rslts = {'model':'linear_regression','coin':coin,'train_size':train_size,'test_size':test_size,'mae':mae,'rmse':rmse}
    rslts_final = pd.DataFrame([rslts])
    return y_true,rslts_final

def model_to_choose(prophet=True,linear_regression=False,dummy_regressor=False):
    if prophet == True:
        print('model selected is prophet')
        rslts = model_prophet()
    if dummy_regressor == True:
        print('You will be using a dummy regressor')
        rslts=dummy_regressor()
    if linear_regression == True:
        print('model is a linear regression model')
        rslts=linear_regressor()
    return rslts
