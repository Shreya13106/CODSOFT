# src/predict.py
import pandas as pd
import joblib

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Predict survival for a single passenger
    """
    # Load model and scaler
    model = joblib.load('../models/random_forest_tuned_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    
    # Create passenger dataframe
    passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [0 if sex.lower() == 'female' else 1],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [{'c':0, 'q':1, 's':2}[embarked.lower()]],
        'Title': [1],
        'FamilySize': [sibsp + parch + 1],
        'IsAlone': [1 if (sibsp + parch) == 0 else 0]
    })
    
    # Scale and predict
    passenger_scaled = scaler.transform(passenger)
    prediction = model.predict(passenger_scaled)[0]
    probability = model.predict_proba(passenger_scaled)[0][1]
    
    result = "Survived" if prediction == 1 else "Did not survive"
    return result, probability

# Example usage
if __name__ == "__main__":
    result, prob = predict_survival(1, 'female', 22, 0, 0, 150, 'C')
    print(f"Prediction: {result}")
    print(f"Probability: {prob:.2%}")