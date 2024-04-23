# Import necessary functions and variables
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Define tokenizer and max_length variables
tokenizer = Tokenizer()
max_length = 100  # Example value, please replace with your actual max length

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

# Ensure correct order of function definitions
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

# Run the main function
if __name__ == "__main__":
    main()
