import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the trained model
model = load_model(r'C:\Users\DELL\Downloads\AI_LeadTech\MLOPS\streamlit_NLP\model01.h5')

# Load the saved tokenizer used during training
with open(r'C:\Users\DELL\Downloads\AI_LeadTech\MLOPS\streamlit_NLP\tokenizer01.pkl', 'rb') as tk:
    tokenizer = pickle.load(tk)

# Define function to preprocess user text input
def preprocess_text(text):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])

    # Pad the sequences to a fixed lenght
    padded_tokens = pad_sequences(tokens, maxlen = 100)
    return padded_tokens[0]

# Create the Title of the APP
st.title('Twitter US Airline Sentiment')


# Create text input widget for user input
user_input = st.text_area('Enter text for sentiment analysis', ' ')

# Create a Button to trigger the sentiment analysis
if st.button('Predict Sentiment'):
    # Preprocess the user input
    preprocess_input = preprocess_text(user_input)

    # Make prediction using the loaded module
    prediction = model.predict(np.array([preprocess_input]))

    # Assuming prediction[0] is a list or array containing the model's output probabilities for each class
    prediction_values = prediction[0]

    # Use argmax to find the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction_values)

    # Define the sentiment classes
    sentiment_classes = ['Neutral', 'Positive', 'Negative']

    # Get the sentiment label based on the predicted class index
    sentiment = sentiment_classes[predicted_class_index]

        
    
    st.write(prediction)

    # Display the sentiment
    st.write(f' ### Sentiment is: {sentiment}')
