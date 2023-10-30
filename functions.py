import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.dummy import DummyRegressor
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from pandas import to_datetime
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


     # create two columns for charts
#     fig_col3, fig_col4 = st.columns(2)

#     with fig_col3:
#         st.markdown("### Weekly Moving Average") 
        
#         new_df = df1[df1['coin'] == select_param]

#         new_df['moving_average_7'] = new_df['price_in_usd'].rolling(7).mean()
#         new_df.dropna(inplace=True)
#         fig3 = px.line(data_frame=new_df, y = 'moving_average_7',x=pd.to_datetime(new_df['date']),
#             title = 'Weekly Moving Average')
#         fig3.update_layout(xaxis_title='date',yaxis_title='Price in USD')
        
#         st.write(fig3)
        
#     with fig_col4:
#         st.markdown("### Monthly Moving Average")  
          
#         new_df = df1[df1['coin'] == select_param]
#         new_df['moving_average_30'] = new_df['price_in_usd'].rolling(30).mean()
#         new_df.dropna(inplace=True)

#         fig4 = px.line(data_frame=new_df, y = 'moving_average_30',x=pd.to_datetime(new_df['date']),
#             title = 'Monthly Moving Average')
#         fig4.update_layout(xaxis_title='date',yaxis_title='Price in USD')
        
#         st.write(fig4)
        