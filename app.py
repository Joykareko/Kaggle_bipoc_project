import pandas as pd
import plotly.express as px
import streamlit as st
import datetime
from pandas import to_datetime
from models import model_to_choose,model_prophet,linear_regressor,dummy_regressor


def set_page_config():
    st.set_page_config(
        page_title="Cryptometrics",
        page_icon=":bar_chart:",
        layout="wide", 
        initial_sidebar_state="expanded",
    )
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)
    

    
#@st.experimental_memo  #ensures the data is loaded once
def load_data() -> pd.DataFrame:
    data = pd.read_csv('crypto_upto_date_prices/combined_file.csv')
    data_2 = pd.read_csv('crypto_upto_date_prices/combined_file_market_cap.csv')
    return data,data_2


    
        
def main():
    set_page_config()

    df1,df2 = load_data()

    st.title("📊 Cryptometrics Dashboard")
    select_param = st.container()
    
    with select_param:
        param_lst = list(df1['coin'].unique())
        select_param = st.selectbox('Select a Coin', param_lst)
        
    #tab1,tab2 = st.tabs(['Descriptives','Predictions'])
    
    b1, b2, b3, b4 = st.columns(4)
    
    with b1:
        st.markdown('Name of Coin')
        st.write(select_param)
    with b2:
        st.markdown('Est Year')
        new_df=df1[df1.coin==select_param]
        st.write((new_df['date'].head(1)).iloc[-1])
    
    with b3:
        st.markdown('Current Price(USD)')
        new_df2=df1[df1.coin==select_param]
        price = (new_df2['price_in_usd'].tail(1)).iloc[-1]
        st.write(str(round(price)))
        
    with b4:
        st.markdown('Current Market Capitalization(USD)')
        new_df3=df2[df2.coin==select_param]
        mark_cup = f"{(new_df3['market_capitalization_in_usd'].tail(1)).iloc[-1]:,}"
        st.write(str(mark_cup))
        
   

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.markdown("### Prices Trend")
        df1 = df1[df1.coin==select_param]
        fig = px.line(
            data_frame=df1, x= pd.to_datetime(df1['date']), y="price_in_usd",
              title='Prices Trend in USD')
        
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.update_layout(xaxis_title='Year',yaxis_title='Price in USD')
        
        st.write(fig)

    with fig_col2:
        st.markdown("### Market Capitalization Trend")
        df2 = df2[df2.coin==select_param]
        fig2 = px.line(
            data_frame=df2, x = pd.to_datetime(df2['date']),y='market_capitalization_in_usd',
        title = 'Market Capitalization in USD')
        
        fig2.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig2.update_layout(xaxis_title='Year',yaxis_title='Market Capitalization in USD')
        st.write(fig2)
         # create two columns for charts
    fig_col3, fig_col4 = st.columns(2)

    with fig_col3:
        st.markdown("### Weekly Moving Average") 
        
        new_df = df1[df1['coin'] == select_param]

        new_df['moving_average_7'] = new_df['price_in_usd'].rolling(7).mean()
        new_df.dropna(inplace=True)
        fig3 = px.line(data_frame=new_df, y = 'moving_average_7',x=pd.to_datetime(new_df['date']),
            title = 'Weekly Moving Average')
        fig3.update_layout(xaxis_title='date',yaxis_title='Price in USD')
        
        st.write(fig3)
        
    with fig_col4:
        st.markdown("### Monthly Moving Average")  
          
        new_df = df1[df1['coin'] == select_param]
        new_df['moving_average_30'] = new_df['price_in_usd'].rolling(30).mean()
        new_df.dropna(inplace=True)

        fig4 = px.line(data_frame=new_df, y = 'moving_average_30',x=pd.to_datetime(new_df['date']),
            title = 'Monthly Moving Average')
        fig4.update_layout(xaxis_title='date',yaxis_title='Price in USD')
        
        st.write(fig4)
        
    #fig5 = st.columns(1) 
    
    select_param2 = st.container()
    select_model = st.container()
    model_to_choose = st.container()
    number_of_days = st.container()
    number_of_training_days = st.container()
    #with fig5:
    st.markdown("#### Make predictions to your favourite coin")
    param_lst2 = list(df1['coin'].unique())
    select_param2 = st.selectbox('Select a Coin', param_lst2)
    model_to_choose = st.selectbox('Select a model',['Prophet','Dummy Regressor','Linear Regressor'])
    number_of_days = st.selectbox('Select number of days you wwant to predict',[7])
    number_of_training_days = st.selectbox('Select number of training days',[365,180,30])
    y_true,table  = linear_regressor()
    st.write(y_true)
    st.write(table)
            
        #st.write(fig5)
if __name__ == '__main__':
    main()        