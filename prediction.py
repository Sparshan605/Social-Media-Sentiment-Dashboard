import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def predict(text_data):
    model = joblib.load("output_models/logistic_regression_model.sav")
    max_features = min(5000, max(100, len(set(' '.join(text_data).split()))))
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_transformed = vectorizer.fit_transform(text_data)
    return model.predict(X_transformed)