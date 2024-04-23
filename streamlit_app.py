import streamlit as st
import tensorflow as tf

# Load the model
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
