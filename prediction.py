import joblib

def predict(data):
    model=joblib.load("output_models/logistic_regression_model.sav")
    return model.predict(data)
    