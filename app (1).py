import streamlit as st
import pickle
import re

# Function to clean text (should be the same as used during training)
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load the saved model, vectorizer, and label encoder
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open('label_encoder.pkl', 'rb') as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)
    st.success("Models, vectorizer, and label encoder loaded successfully!")
except FileNotFoundError:
    st.error("Error: One or more necessary files (model, vectorizer, or label encoder) not found. Please ensure they are saved in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# Streamlit app layout
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (positive/negative).")

user_input = st.text_area("Enter your movie review here:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_input = clean_text(user_input.lower())
        # Vectorize the cleaned text
        vectorized_input = vectorizer.transform([cleaned_input])

        # Make prediction
        prediction_numeric = model.predict(vectorized_input)
        # Convert numerical prediction back to original label
        predicted_sentiment = label_encoder.inverse_transform(prediction_numeric)[0]

        st.write(f"**Predicted Sentiment:** {predicted_sentiment.capitalize()}")

        if predicted_sentiment == 'positive':
            st.balloons()
        else:
            st.snow()
    else:
        st.warning("Please enter some text to analyze.")
