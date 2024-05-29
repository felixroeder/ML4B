import streamlit as st
import datetime
from datetime import date
import tensorflow as tf
import pandas as pd
from nltk.corpus import stopwords
import string

#from folder1.QuellCode import create_new_price

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
    #cst.success(create_new_price(st.session_state.news, st.session_state.price))
    st.session_state.submitted = False
    
    # Zusätzlicher Slider für Investitionspräferenz
    with st.form(key='my_form1'):
        Ranking = st.number_input(label='Wie gerne investieren Sie am Aktienmarkt?', max_value = 10, min_value = 1)
        submit_button = st.form_submit_button(label='Submit')

class Model:
    data = pd.read_csv("folder1/googletest.csv", encoding="utf-8", delimiter=";")
    date_col = "Date"  # Column containing the date
    price_col = "Close"  # Column containing the closing price
    news_col = "News_Article"  # Column containing the news text (optional)

    print("Daten  ", data)
    # Prepare data
    data['shifted_price'] = data['Close'].shift(-1)  # Create new column with shifted price
    data.dropna(inplace=True)  # Remove rows with missing values

    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Text Preprocessing (if using news articles)
    def preprocess_text(text):

        # Lowercase text
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords
        stop_words = stopwords.words('english')
        text = ' '.join([word for word in text.split() if word not in stop_words])

        return text

    # Apply preprocessing to each headline
    train_news = train_data[news_col].apply(preprocess_text)
    test_news = test_data[news_col].apply(preprocess_text)

    # Text Vectorization (if using news articles)
    max_vocab_size = 100000  # Adjust based on your data
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_vocab_size, output_sequence_length=391, output_mode = 'int')  # Set output_sequence_length
    vectorizer.adapt(train_news.tolist() + test_news.tolist())
