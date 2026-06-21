# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    # Load data
    df = pd.read_csv('../data/Titanic-Dataset.csv')
    
    # Clean data
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Feature engineering
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].fillna('Unknown')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Encode
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    
    # Prepare features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    X = df[features]
    y = df['Survived']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Model Accuracy: {accuracy:.2%}")
    
    # Save model and scaler
    joblib.dump(model, '../models/random_forest_tuned_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_model()