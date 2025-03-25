import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def predict(text_data):
    # Load the pre-trained model and vectorizer
    model = joblib.load("output_models/logistic_regression_model.sav")
    vectorizer = joblib.load("output_models/tfidf_vectorizer.sav")
    
    try:
        # Transform the input text data using the EXACT same vectorizer used during training
        X_transformed = vectorizer.transform(text_data)
        
        # Make predictions
        predictions = model.predict(X_transformed)
        
        return predictions
    
    except ValueError as e:
        print(f"Prediction error: {e}")
        return ['unable_to_predict'] * len(text_data)