import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    # Load JSON model architecture
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # Load weights into new model
    model.load_weights("model.h5")  # Update model file name
    return model

# Load the model
model = load_model()

# Streamlit UI
st.title("Hate Speech Detection")

# Text input area
text_input = st.text_area("Enter text:")

# Button to make predictions
if st.button("Detect Hate Speech"):
    if text_input:
        # Preprocess input text
        preprocessed_text = preprocess(text_input)  # Preprocess the input text
        
        # Make predictions
        prediction = model.predict(preprocessed_text)
        
        # Display prediction result
        st.write("Prediction:", prediction)
    else:
        st.warning("Please enter some text.")
