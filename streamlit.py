import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Update model file name
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
        # Preprocess input text and make predictions
        # (Your preprocessing steps here)
        prediction = model.predict(preprocessed_text)
        
        # Display prediction result
        st.write("Prediction:", prediction)
    else:
        st.warning("Please enter some text.")
