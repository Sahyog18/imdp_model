import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Data Loading ---
print("Loading data...")
df = pd.read_csv('/content/IMDB Dataset.csv')

# --- 2. Initial Data Cleaning ---
print("Cleaning data...")
df.drop_duplicates(inplace=True)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['review'] = df['review'].apply(clean_text)

# --- 3. Text Vectorization ---
print("Vectorizing text using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['review'])

# --- 4. Label Encoding ---
print("Encoding sentiment labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# --- 5. Data Splitting ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Model Training and Hyperparameter Tuning (Logistic Regression) ---
print("Starting GridSearchCV for Logistic Regression...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

log_reg = LogisticRegression(max_iter=1000, random_state=42)
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_log_reg_model = grid_search.best_estimator_
print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# --- 7. Model Evaluation ---
print("\nEvaluating the best model on the test set...")
y_pred_tuned_log_reg = best_log_reg_model.predict(X_test)
print("Tuned Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_tuned_log_reg))
print("\nTuned Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_tuned_log_reg))

# --- 8. Saving Components ---
print("\nSaving model, vectorizer, and label encoder...")
with open('model.pkl', 'wb') as file:
    pickle.dump(best_log_reg_model, file)
print("Saved model.pkl")

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
print("Saved tfidf_vectorizer.pkl")

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
print("Saved label_encoder.pkl")

print("Model building process complete.")
