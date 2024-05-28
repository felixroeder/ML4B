import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import requests
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from textblob import TextBlob

# Load historical data (replace with your data loading logic)

data = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/googletest.csv", encoding="utf-8", delimiter=";")
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

train_news_sequences = vectorizer(train_news.tolist())
test_news_sequences = vectorizer(test_news.tolist())

# Ensure the news sequences are correctly shaped as integer indices
train_news_sequences = np.array(train_news_sequences)
test_news_sequences = np.array(test_news_sequences)

# Define look-back window
look_back = 5  # Number of past days (including news) to consider for prediction

# Sentiment Analysis 
def get_binary_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return 1 if polarity >= 0 else 0

train_sentiment = train_news.apply(get_binary_sentiment).astype(float)
test_sentiment = test_news.apply(get_binary_sentiment).astype(float)

# Create sequences with a look-back window
def create_sequences(news, price, sentiment, window_size):
    sequences, labels = [], []
    for i in range(len(price) - window_size):
        news_seq = news[i:i + window_size]
        sentiment_seq = sentiment[i:i + window_size]
        combined_seq = np.hstack([news_seq, np.array(sentiment_seq).reshape(-1, 1)])        
        sequences.append(combined_seq)
        labels.append(price[i + window_size])  # The target price is the next price after the window
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

train_sequences, train_labels = create_sequences(train_news_sequences, train_data[price_col].values, train_sentiment.values, look_back)
test_sequences, test_labels = create_sequences(test_news_sequences, test_data[price_col].values, test_sentiment.values, look_back)

input_dim = train_sequences.shape[-1]  # The number of features in each sequence# Build the model

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(look_back, input_dim)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

#assert train_sequences.dtype == 'int32' or train_sequences.dtype == 'float32', "Train sequences must be int32 or float32"
#assert train_labels.dtype == 'float64' or train_labels.dtype == 'float32', "Train labels must be float32 or float64"

# Compile model
model.compile(loss="mse", optimizer="adam", run_eagerly = True)

# Train the model
model.fit(train_sequences, train_labels, epochs=10, batch_size=32)

# Make predictions on test data
predicted_price = model.predict(test_sequences)
print(predicted_price[0])

#####################Neue Nachrichten einbauen

# Define your API key and endpoint
api_key = '5a9f2b08c11f43b0a3fcde731a6b8707'
endpoint = 'https://newsapi.org/v2/everything'

# Parameters for the API request
params = {
    'q': 'finance',
    'apiKey': api_key,
    'language': 'en',
    'sortBy': 'publishedAt'
}

# Function to fetch news and return a DataFrame
def fetch_financial_news(api_endpoint, parameters):
    response = requests.get(api_endpoint, params=parameters)
    news_data = response.json()

    articles = news_data.get('articles', [])
    
    # Extract headlines and publication dates
    data = [(article['title'], article['publishedAt']) for article in articles]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Headline', 'Date'])
    
    # Convert 'Date' from string to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

###bzw. bei existierendem DataFrame (wie in unserem Fall)
existing_df = pd.DataFrame(columns=['Headline', 'Date'])  # Or load from an existing file

# Fetch new financial news
new_news_df = fetch_financial_news(endpoint, params)

# Append new rows to the existing DataFrame
updated_df = pd.concat([existing_df, new_news_df], ignore_index=True)

# Optionally, save the updated DataFrame to a CSV file
updated_df.to_csv('financial_news.csv', index=False)

# Function to fetch stock data and return a DataFrame
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Reset the index to make 'Date' a column
    data.reset_index(inplace=True)
    
    # Keep only the relevant columns
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return data

# Parameters
ticker_symbol = 'AAPL'  # Example ticker symbol for Apple Inc.
start_date = '2022-01-01'
end_date = '2022-12-31'

#bzw. bei existierendem DataFrame
existing_stock_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])  # Or load from an existing file

# Fetch new stock data
new_stock_data_df = fetch_stock_data(ticker_symbol, start_date, end_date)

# Append new rows to the existing DataFrame
updated_stock_df = pd.concat([existing_stock_df, new_stock_data_df], ignore_index=True)

# Optionally, save the updated DataFrame to a CSV file
updated_stock_df.to_csv('stock_prices.csv', index=False)

# Use the model for future predictions (replace with your new data)
new_news = preprocess_text("Your new news article")  # Preprocess new news article
new_news_sequence = vectorizer(np.array([new_news]))
new_price_data = data[price_col]

# Include logic for new price data (replace with your actual approach)
# Assuming you want to use the most recent closing price
new_price_data = np.array([[data[price_col].iloc[-1]]])  # Access the last closing price

# Combine new news and price data
new_data = {
    "news": new_news_sequence,
    "price": new_price_data
}

new_sentiment = get_binary_sentiment(new_news)

# Create a sequence from the new data
new_sequence = create_sequences(new_news_sequence, new_price_data, new_sentiment, look_back)
new_sequence = np.array(new_sequence)

# Predict the future price using the trained model
# NOCH KEINE NEUTEN DATEN EINGEPFLEGT #predicted_price = model.predict(new_sequence)  # Access the first element from the prediction

#print(f"Predicted future price: {predicted_price}")
