import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from imblearn.over_sampling import SMOTE

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load dataset and preprocess
def preprocess_data():
    # Load the dataset, skip bad lines
    df = pd.read_csv('C:/Users/nithish/Desktop/email_spam_detection/dataset/spam_data.csv', on_bad_lines='skip')
    
    # Print the column names to check for any discrepancies
    print(df.columns)

    # Basic text cleaning
    df['text'] = df['text'].str.lower()  # Convert to lowercase
    df['text'] = df['text'].str.replace(r'\d+', '')  # Remove numbers
    df['text'] = df['text'].str.replace(r'[^\w\s]', '')  # Remove punctuation
    
    # Save the cleaned dataset to a new file
    df.to_csv('C:/Users/nithish/Desktop/email_spam_detection/dataset/spam_data_cleaned.csv', index=False)
    
    # Vectorization with TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['text']).toarray()
    y = df['label']
    
    # Apply SMOTE for balancing the dataset (oversample minority class)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled, tfidf

# Train the Logistic Regression model
def train_model():
    X, y, tfidf = preprocess_data()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model and TF-IDF vectorizer
    joblib.dump(model, 'C:/Users/nithish/Desktop/email_spam_detection/backend/model/logistic_model.pkl')
    joblib.dump(tfidf, 'C:/Users/nithish/Desktop/email_spam_detection/backend/model/tfidf_vectorizer.pkl')
    
    # Print model performance on the test set
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Initial training (only run once when setting up the project)
train_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("text", "")
    
    # Load the saved model and TF-IDF vectorizer
    model = joblib.load('C:/Users/nithish/Desktop/email_spam_detection/backend/model/logistic_model.pkl')
    tfidf = joblib.load('C:/Users/nithish/Desktop/email_spam_detection/backend/model/tfidf_vectorizer.pkl')
    
    # Preprocess and vectorize the input text
    email_vec = tfidf.transform([email_text]).toarray()
    
    # Predict using the trained model
    prediction = model.predict(email_vec)
    response_message = "Spam" if prediction[0] == 1 else "Not Spam"
    
    return jsonify({"message": response_message})

if __name__ == "__main__":
    app.run(debug=True)
