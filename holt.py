# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:30:05 2020

@author: gauravrai
"""

import datetime as dt
import nsepy
from statsmodels.tsa.api import Holt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import pathlib
DATA_PATH = pathlib.Path(__file__).parent.joinpath("data").resolve()

# We would like all available data from 01/01/2000 until 12/31/2016.
share_list=pd.read_csv(DATA_PATH.joinpath("EQUITY_L.csv"))
today = dt.date.today()
yesterday = today-dt.timedelta(days=10)

st.markdown("<H1 style='text-align:left;color: #B51B5C;'><b> Holt Winter Based Share Price Prediction Model</b></H1>",unsafe_allow_html=True)
st.sidebar.header("Share Price Prediction Model")
#'''-----------------Date Selection--------------'''
start_date = st.sidebar.date_input('Start date', yesterday)
end_date = st.sidebar.date_input('End date',today)
dates=[]
delta = end_date - start_date       # as timedelta
for i in range(delta.days + 1):
    day = start_date + dt.timedelta(days=i)
    dates.append(day)
if start_date > end_date:
    st.error('Error: End date must fall after start date.')

#'''---------------Drop Down Menu--------------------'''    
stock=st.sidebar.selectbox("Select the Equity listed on NSE",share_list['SYMBOL'])
data = nsepy.get_history(symbol=stock,start=start_date,end=end_date)
col=data.columns.tolist()
@st.cache(ttl=3600*24, show_spinner=False)
def load_data():
    share_list=pd.read_csv(DATA_PATH.joinpath("EQUITY_L.csv"))
    data = nsepy.get_history(symbol=stock,start=start_date,end=end_date)
    return share_list,data
share_list,data =load_data()
#'''------------Candle Stick Graph--------------------------'''

candle_graph= make_subplots(rows=2, cols=1,shared_yaxes=False,shared_xaxes=True,vertical_spacing = 0.01,row_heights=[500,100])
candle_graph.add_trace(go.Candlestick(x=dates,open=data['Open'], high=data['High'],low=data['Low'], close=data['Close'],name='share price'),row=1, col=1)

candle_graph.add_trace(go.Bar(x=dates,y=data['Volume'].values.tolist(),name='Share_Volume'),row=2, col=1)                     

candle_graph.update_layout(xaxis_rangeslider_visible=False,
    title={
        'text': "<b>Stock Price Chart of {}</b>".format(stock),
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},    
        font=dict(
        family="Courier New, monospace",
        size=20,
        color="RebeccaPurple"),width=1200, height=600)
candle_graph.update_yaxes(showticklabels=False)
st.write(candle_graph)

def HLM_model(train,days):
    #alpha=smoothing_level and beta=smoothing slope
    fit1 = Holt(train).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    fcast=fit1.forecast(days)
    pdates=[]
    for i in range(days + 1):
        day = end_date + dt.timedelta(days=i)
        pdates.append(day)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pdates, y=fcast, name='Predicted Price'))
    fig.update_layout(    
        title={
        'text': "<b>Predicted Stock Price Chart of {}</b>".format(stock),
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},    
        font=dict(
        family="Courier New, monospace",
        size=20,
        color="RebeccaPurple"))
    st.write(fig)
    
pred_days=st.sidebar.slider("Select No of days for which price need to be predicted",min_value=1,max_value=30,step=1)
HLM_model(data['Close'],pred_days)
st.sidebar.image('gaurav.jpg',caption= "Gaurav Rai")
st.sidebar.info("I have more then 2 years of experience in data science and web app development with specialization in Time Series Analysis....................You can connect with me on email id: gauravcs08@gmail.com")