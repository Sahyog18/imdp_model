import streamlit as st
import pickle
import nltk
import re
import string
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model and vectorizer
# Ensure these files exist from the previous training step
if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
else:
    st.error("Model or Vectorizer files not found. Please ensure they are trained and saved.")
    st.stop() # Stop the app if files are missing

# Text cleaning function (as defined previously)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Changed to raw string for \S+
    text = re.sub(r'\d+', '', text) # Changed to raw string for \d+
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Streamlit UI
st.title("IMDB Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):
    if review:
        processed_review = clean_text(review)
        # Ensure the vectorizer has been fitted with vocabulary before transforming
        if hasattr(vectorizer, 'vocabulary_') and vectorizer.vocabulary_:
            review_vector = vectorizer.transform([processed_review])
            prediction = model.predict(review_vector)

            if prediction[0] == 1:
                st.success("Positive Review 😊")
            else:
                st.error("Negative Review 😔")
        else:
            st.warning("Vectorizer has no vocabulary. Please ensure it was fitted with data.")
    else:
        st.warning("Please enter a review to predict.")
