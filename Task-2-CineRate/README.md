# Task-2: CineRate - Movie Rating Prediction System

A Machine Learning project that predicts IMDb ratings for Indian movies based on genre, director, cast, duration, votes, and release year.

---

## Overview

CineRate is an end-to-end Data Science project that analyzes movie data and predicts IMDb ratings using Machine Learning. The project includes data preprocessing, exploratory data analysis, feature engineering, model training, and deployment via a Streamlit web application.

---

## Objectives

- Analyze IMDb movie data to identify factors affecting ratings
- Build and compare multiple regression models
- Deploy the best model as an interactive web application
- Provide real-time movie rating predictions

---

## Features

- Complete data cleaning and preprocessing pipeline
- Professional visualizations and insights
- 5 Machine Learning models trained and compared
- Interactive Streamlit web application
- Real-time rating predictions with visual feedback

---

## Dataset

**Source:** [IMDb India Movies Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)

**Features:**
- Name, Year, Duration, Genre, Rating, Votes, Director, Actors

**Size:** 1000+ Indian movies

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming |
| Pandas, NumPy | Data processing |
| Matplotlib, Seaborn | Visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Gradient Boosting |
| Streamlit | Web application |
| Joblib | Model serialization |

---

## Machine Learning Models

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Random Forest**  | **0.71** | **0.52** | **0.68** |
| XGBoost | 0.70 | 0.53 | 0.69 |
| Gradient Boosting | 0.69 | 0.55 | 0.71 |
| Decision Tree | 0.52 | 0.71 | 0.92 |
| Linear Regression | 0.45 | 0.82 | 1.04 |

**Best Model:** Random Forest Regressor

---

## Project Structure

CineRate/
│
├── app/
│   └── streamlit_app_simple.py      # Web application
│
├── data/
│   └── IMDb_India_Movies.csv        # Dataset
│
├── models/
│   ├── best_model.pkl               # Trained model
│   ├── scaler.pkl                   # Feature scaler
│   └── genre_encoder.pkl            # Genre encoder
│
├── notebooks/
│   └── cinerate_no_graphs.py        # Main analysis script
│
├── outputs/
│   └── figures/
│       └── visualizations.png       # Generated charts
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── preprocess.py
│   └── utils.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py

---

## 🔧 Installation

### 1. Clone the Repository

git clone https://github.com/Shreya13106/CineRate.git
cd CineRate


### 2. Create Virtual Environment (Recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) and place in `data/` folder.

---

## How to Run

### Run the Analysis Script
python notebooks/cinerate_no_graphs.py

### Run the Web Application

python -m streamlit run app/streamlit_app_simple.py

### Quick Prediction Test
python -c "
import joblib
import numpy as np

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
genre_encoder = joblib.load('models/genre_encoder.pkl')

# 3 Idiots
duration, votes, year = 170, 400000, 2009
movie_age = 2024 - year
genre = 'Comedy'

genre_encoded = genre_encoder.transform([genre])[0]
features = np.array([[duration, votes, year, movie_age, genre_encoded]])
features_scaled = scaler.transform(features)
rating = model.predict(features_scaled)[0]

print(f'Predicted Rating: {rating:.1f}/10')
"
---

## Sample Predictions

| Movie | Duration | Votes | Year | Genre | Predicted Rating |
|-------|----------|-------|------|-------|------------------|
| 3 Idiots | 170 min | 400,000 | 2009 | Comedy | 8.2/10 |
| Dangal | 161 min | 300,000 | 2016 | Biography | 8.0/10 |
| RRR | 180 min | 500,000 | 2022 | Action | 7.8/10 |
| Drishyam | 163 min | 100,000 | 2015 | Thriller | 8.1/10 |

---

## Key Findings

- **Votes** have the strongest correlation with ratings (0.42)
- **Director success score** significantly impacts ratings
- **Comedy and Drama** genres tend to receive higher ratings
- Movies between **120-180 minutes** perform best

---

## Future Improvements

- Add more features (budget, production house)
- Implement Deep Learning models
- Add sentiment analysis from reviews
- Deploy to cloud (AWS/GCP/Azure)
- Create REST API for predictions

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- Kaggle for the dataset
- Streamlit for the web framework
- Scikit-learn for ML implementations
- Open source community
