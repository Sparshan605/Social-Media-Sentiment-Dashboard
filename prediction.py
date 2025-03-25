import joblib
import numpy as np

def predict(data):
    
    model = joblib.load("output_models/logistic_regression_model.sav")
    vectorizer = joblib.load("output_models/tfidf_vectorizer.sav")
    
    
    transformed_data = vectorizer.transform(data)
    
    return model.predict(transformed_data)