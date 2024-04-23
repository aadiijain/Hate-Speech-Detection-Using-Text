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
