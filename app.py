
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
import streamlit as st

## Load trained model

model = load_model('/content/drive/MyDrive/DEEP LEARNING NLP/SimpleRNN/simple_rnn_model_imdb.h5')

# Compile Model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuraccy']) 

## Load word index from imdb
word_index = imdb.get_word_index()
reversed_word_index = {value:key for key,value in word_index.items()}

## Streamlit UI

st.title("🎬 IMDB Movie Review Sentiment Analysis") # Title

st.write("Enter a movie review to predict whether it is Positive or Negative")

review = st.text_area("Movie Review") # User Input

## Preprocess function

def preprocess_review(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2)+3 for word in words]
  
  # Padding the review
  padded_review = sequence.pad_sequences([encoded_review],maxlen=500)

  return padded_review

## Prediction 

if st.button("Predict Sentiment"):

  if review.strip() == "":
    st.warnin("Please enter a movie review")
  
  else:
    preprocessed_review = preprocess_review(review)

    prediction = model.predict(preprocessed_review)

    prediction_score = prediction[0][0]

    st.write("Prediction Score:",prediction_score)

    if prediction_score > 0.5:
      st.success("😊 Positive Review")
    else:
      st.error("😞 Negative Review")
