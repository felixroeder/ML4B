
import streamlit as st
import datetime
from datetime import date
#from datei import Code


# Titel der App
st.title("Stock Prediction App")
st.subheader('This app is created to forecast the stock market movement using daily news headlines')
st.image("https://t4.ftcdn.net/jpg/02/20/32/75/240_F_220327557_gRDTuYL4iVG0lWrjgjrv1chBCUunjKlG.jpg")


# Zustand des Formulars initialisieren
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Formular erstellen und anzeigen, wenn es noch nicht abgesendet wurde
if not st.session_state.submitted:
    with st.form(key='my_form'):
        news_headlines = st.text_input(label='Enter News Headline(s), the more the better ;)')
        price_yesterday = st.number_input(label='Yesterdays closing Stock Price (currently Google [Alphabet] only)')
        submit_button = st.form_submit_button(label='Submit')
    
    # Wenn das Formular abgesendet wurde, den Zustand ändern
    if submit_button:
        st.session_state.submitted = True
        st.session_state.news = news_headlines
        st.session_state.price = price_yesterday

# Zeigen Sie die Erfolgsmeldung und den Slider, nachdem das Formular abgesendet wurde
if st.session_state.submitted:
    st.success('Hier würde jetzt der morgige Kurs stehen wenn das Modell fertig wäre')

    # Zusätzlicher Slider für Investitionspräferenz
    invest_preference = st.slider('Wie gerne investieren Sie am Aktienmarkt? (1-10)', 1, 10, 5)
    st.write(f'Danke für die Teilnahme')