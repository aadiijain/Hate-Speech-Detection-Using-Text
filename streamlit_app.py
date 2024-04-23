import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import pickle  # Add import for pickle module

# Download NLTK stopwords
nltk.download('stopwords')

# Load the model architecture from JSON file
try:
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # Load the model weights
    model.load_weights("model.h5")

    # Load the tokenizer
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)

    max_length = model.input_shape[1]

    # Function to preprocess the input text
    def preprocess_text(text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\@\w+', '', text)  # Remove @mentions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove all non-alphanumeric and non-space characters
        text = ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])  # Remove stop words
        return text

    # Function to predict the class
    def predict_class(text):
        text = preprocess_text(text)
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        prediction = model.predict(padded_sequences)
        return prediction.argmax(axis=-1)[0]

    # Streamlit app
    def main():
        st.title("Hate Speech Detection")
        st.write("Enter text to detect if it contains hate speech or offensive language.")
        input_text = st.text_input("Enter text:")
        if st.button("Predict"):
            prediction = predict_class(input_text)
            if prediction == 0:
                st.write("Predicted Class: Hate Speech")
            elif prediction == 1:
                st.write("Predicted Class: Offensive Language")
            else:
                st.write("Predicted Class: Neither")

    if __name__ == "__main__":
        main()

except FileNotFoundError:
    st.error("Error: Required files not found. Make sure model.json, model.h5, and tokenizer.pickle are present.")
