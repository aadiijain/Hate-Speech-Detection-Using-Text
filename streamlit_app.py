import streamlit as st
import tensorflow as tf
import json

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

# Function to perform inference
import numpy as np

# Function to perform inference
def predict_hate_speech(model, text):
    try:
        # Preprocess the input text
        # Example: Tokenization and padding
        tokenizer = YourTokenizerClass()  # Initialize tokenizer based on your preprocessing method
        max_sequence_length = YourMaxLengthValue  # Set the maximum sequence length based on your model

        # Tokenize the text and pad sequences
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

        # Perform inference
        prediction = model.predict(padded_sequences)

        # Assuming your model outputs probabilities or logits,
        # you can decide a threshold to classify as positive or negative
        threshold = 0.5
        if prediction >= threshold:
            return "Positive"
        else:
            return "Negative"

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"  # Return an error message or handle the exception as needed

# Run the main function
if __name__ == "__main__":
    main()
