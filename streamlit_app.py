import streamlit as st
import tensorflow as tf

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Update model file name if needed
    return model

# Preprocessing function
def preprocess(text):
    preprocessed_text = text.lower()  # Convert text to lowercase
    return preprocessed_text

# Main function to run the Streamlit app
def main():
    # Load the model
    model = load_model()

    # Title of the app
    st.title("Hate Speech Detection")

    # Input text box
    text_input = st.text_input("Enter text:")

    # Preprocess the input text
    preprocessed_text = preprocess(text_input)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(preprocessed_text)  # Update this part based on your model input format
        st.write("Prediction:", prediction)  # Display the prediction result

# Run the app
if __name__ == "__main__":
    main()
