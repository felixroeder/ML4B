
import streamlit as st
import datetime
from datetime import date
from datei import Code


st.title("Stock Prediction App")
st.subheader('This app is created to forecast the stock market movement using daily news headlines')
st.image("https://t4.ftcdn.net/jpg/02/20/32/75/240_F_220327557_gRDTuYL4iVG0lWrjgjrv1chBCUunjKlG.jpg")

st.session_state.News = ""
st.session_state.Price = ""

with st.form(key="News and stock data"):
    news_headline = st.text_input("Enter News Headline(s)")
    stockprice_yesterday = st.text_input("Yesterdays closing Stock Price (currently Google only)")
    submitted = st.form_submit_button("Submit")

if submitted:
    st.session_state.News = news_headline
    st.session_state.Price = stockprice_yesterday

def output(preis):
    st.write(preis)

data_load_state = st.text("Load data...")
