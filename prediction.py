import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def predict(text_data):
    model = joblib.load("output_models/logistic_regression_model.sav")
    vectorizer = TfidfVectorizer(max_features=27111)
    X_transformed = vectorizer.fit_transform(text_data)
    return model.predict(X_transformed)