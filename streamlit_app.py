import nltk
nltk.download('stopwords')
import streamlit as st
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras import backend as K
import re

def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Load model architecture from JSON file
        with open("model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = tf.keras.models.model_from_json(loaded_model_json)

        # Load weights into the new model
        model.load_weights("model.h5")

        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Main function to run the Streamlit app
def main():
    # Title of the app
    st.title("Hate Speech Detection")

    # Load the model
    model = load_model()

    if model:
        # Input text area for user input
        text_input = st.text_area("Enter text:", "")

        # Button to trigger inference
        if st.button("Detect Hate Speech"):
            # Perform inference
            prediction = predict_hate_speech(model, text_input)
            st.write("Prediction:", prediction)

# Function to preprocess the input text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stop_words.add("rt")
    
    # Preprocessing steps
    text = text.lower()
    text = remove_entity(text)
    text = remove_url(text)
    text = remove_noise_symbols(text)
    text = remove_stopwords(text)
    
    return text

# Function to perform inference
def predict_hate_speech(model, text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize and pad the preprocessed text
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)

    # Perform inference
    prediction = model.predict(padded_sequence)
    
    # Convert prediction to label
    predicted_label = convert_prediction_to_label(prediction)
    
    return predicted_label

# Function to convert prediction to label
def convert_prediction_to_label(prediction):
    # Your code to convert prediction to label goes here
    # For example, if prediction is an array of probabilities, you can use argmax to get the label
    return "Positive"  # Replace this with your actual conversion logic

# Run the main function
if __name__ == "__main__":
    main()
