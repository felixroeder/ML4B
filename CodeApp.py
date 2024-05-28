import streamlit as st
import datetime
from datetime import date

st.title("Stock Prediction App")
st.subheader('This app is created to forecast the stock market movement using daily news headlines')
st.image("https://t4.ftcdn.net/jpg/02/20/32/75/240_F_220327557_gRDTuYL4iVG0lWrjgjrv1chBCUunjKlG.jpg")

st.sidebar.header('Please select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2008, 1, 1))
ent_date = st.sidebar.date_input('End date', date(2016, 12, 31))

lables = ("1", "0")
selected_lables = st.selectbox("please select a lable for prediction", lables)

news_headline = st.text_input("Enter News Headline")

data_load_state = st.text("Load data...")

# Data Visualization
# st.header('Data Visualazation')
# st.subheader('Plot of the data')
# st.write("**Note:** Select the date range on the sidebar, or zoom in the plot and select your specific column")
# fig = px.line(data, x = 'Date', y = data.columns, title = 'Closing price of the stock', width = 1000, height = 600)
# st.plotly_chart(fig)
