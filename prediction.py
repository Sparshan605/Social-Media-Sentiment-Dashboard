import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def predict(text_data, min_features=10, max_features=50000):
    # Load the pre-trained model
    model = joblib.load("output_models/logistic_regression_model.sav")
    
    # Dynamically determine optimal feature count
    unique_words = len(set(' '.join(text_data).split()))
    optimal_features = np.clip(
        unique_words, 
        min_features, 
        max_features
    )
    
    # Create vectorizer with dynamically adjusted features
    vectorizer = TfidfVectorizer(max_features=optimal_features)
    
    # Transform the input text data
    X_transformed = vectorizer.fit_transform(text_data)
    
    # Make predictions
    predictions = model.predict(X_transformed)
    
    return predictions