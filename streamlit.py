import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras import backend as K

# Define custom evaluation metrics
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2*((precisions*recalls)/(precisions+recalls+K.epsilon()))

# Define LSTM model architecture
vocab_size = 10000  # Define your vocabulary size
max_length = 100  # Define your maximum sequence length
output_dim = 200

model = Sequential([
    Embedding(vocab_size, output_dim, input_length=max_length),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax"),  # Assuming 3 classes (0, 1, 2)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1, precision, recall])

# Streamlit UI
st.title("Hate Speech Detection")

# Function to preprocess text input and make predictions
def predict_hate_speech(text):
    # Preprocess text (e.g., tokenization, padding)
    # Use your preprocessing steps here
    
    # Make prediction
    prediction = model.predict(preprocessed_text)
    return prediction

# Text input area
text_input = st.text_area("Enter text:")

# Button to make predictions
if st.button("Detect Hate Speech"):
    if text_input:
        # Preprocess input text and make predictions
        prediction = predict_hate_speech(text_input)
        
        # Display prediction result
        st.write("Prediction:", prediction)
    else:
        st.warning("Please enter some text.")
