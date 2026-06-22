"""
CINERATE - Movie Rating Prediction System
NO GRAPHS VERSION - Saves graphs to files instead of displaying
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Set matplotlib to not show interactive plots
plt.switch_backend('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Get the current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Set paths based on where we are
if 'notebooks' in current_dir:
    # If running from notebooks folder
    data_path = '../data/IMDb_India_Movies.csv'
    models_path = '../models'
    figures_path = '../outputs/figures'
else:
    # If running from project root
    data_path = 'data/IMDb_India_Movies.csv'
    models_path = 'models'
    figures_path = 'outputs/figures'

# Create folders
os.makedirs(models_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

print("="*60)
print("🎬 CINERATE - MOVIE RATING PREDICTION SYSTEM")
print("="*60)

# ============================================
# 1. LOAD DATASET
# ============================================
print("\n[1/8] Loading dataset...")
print(f"Looking for file at: {data_path}")

# Check if file exists
if not os.path.exists(data_path):
    print(f"❌ File not found at {data_path}")
    print("\nLet me check what's in the data folder:")
    if os.path.exists('data'):
        print("Files in ./data:", os.listdir('data'))
    elif os.path.exists('../data'):
        print("Files in ../data:", os.listdir('../data'))
    else:
        print("No data folder found!")
    exit()

# Try different encodings
encodings = ['latin1', 'cp1252', 'utf-8']
df = None

for enc in encodings:
    try:
        df = pd.read_csv(data_path, encoding=enc)
        print(f"✓ Loaded successfully with {enc} encoding")
        break
    except Exception as e:
        print(f"  {enc} failed: {e}")
        continue

if df is None:
    print("❌ Could not load file with any encoding")
    exit()

print(f"✓ Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================
# 2. CLEAN DATA
# ============================================
print("\n[2/8] Cleaning data...")

# Clean Duration
def clean_duration(x):
    if pd.isna(x):
        return None
    import re
    nums = re.findall(r'\d+', str(x))
    return int(nums[0]) if nums else None

df['Duration'] = df['Duration'].apply(clean_duration)
df = df.dropna(subset=['Duration'])
print("  ✓ Duration cleaned")

# Clean Votes
def clean_votes(x):
    if pd.isna(x):
        return 0
    try:
        return int(str(x).replace(',', ''))
    except:
        return 0

df['Votes'] = df['Votes'].apply(clean_votes)
print("  ✓ Votes cleaned")

# Clean Year
def clean_year(x):
    if pd.isna(x):
        return None
    import re
    years = re.findall(r'\d{4}', str(x))
    return int(years[0]) if years else None

df['Year'] = df['Year'].apply(clean_year)
df = df.dropna(subset=['Year'])
df = df[(df['Year'] >= 1900) & (df['Year'] <= 2024)]
print("  ✓ Year cleaned")

# Fill missing values
df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')
df['Actor 1'] = df['Actor 1'].fillna('Unknown')
df['Actor 2'] = df['Actor 2'].fillna('Unknown')
df['Actor 3'] = df['Actor 3'].fillna('Unknown')
df = df.dropna(subset=['Rating'])

print(f"✓ Final dataset: {df.shape[0]} movies")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
print("\n[3/8] Creating features...")

# Get primary genre
df['Primary_Genre'] = df['Genre'].apply(lambda x: x.split(',')[0] if x != 'Unknown' else 'Unknown')

# Movie age
df['Movie_Age'] = 2024 - df['Year']

# Create feature matrix
X = pd.DataFrame()
X['Duration'] = df['Duration']
X['Votes'] = df['Votes']
X['Year'] = df['Year']
X['Movie_Age'] = df['Movie_Age']

# Encode genre
le = LabelEncoder()
X['Genre'] = le.fit_transform(df['Primary_Genre'])

# Target variable
y = df['Rating']

print(f"✓ Features created: {X.shape[1]} features")
print(f"✓ Rating range: {y.min():.1f} - {y.max():.1f}")

# ============================================
# 4. SAVE VISUALIZATIONS (without displaying)
# ============================================
print("\n[4/8] Creating and saving visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Rating Distribution
axes[0,0].hist(df['Rating'], bins=30, edgecolor='black', color='skyblue')
axes[0,0].axvline(df['Rating'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Rating"].mean():.2f}')
axes[0,0].set_title('IMDb Rating Distribution', fontweight='bold')
axes[0,0].set_xlabel('Rating')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# 2. Top Genres
genre_counts = df['Primary_Genre'].value_counts().head(10)
axes[0,1].barh(range(len(genre_counts)), genre_counts.values, color='lightcoral')
axes[0,1].set_yticks(range(len(genre_counts)))
axes[0,1].set_yticklabels(genre_counts.index)
axes[0,1].set_title('Top 10 Genres', fontweight='bold')
axes[0,1].set_xlabel('Number of Movies')

# 3. Genre Ratings
genre_ratings = df.groupby('Primary_Genre')['Rating'].mean().sort_values(ascending=False).head(10)
axes[0,2].barh(range(len(genre_ratings)), genre_ratings.values, color='lightgreen')
axes[0,2].set_yticks(range(len(genre_ratings)))
axes[0,2].set_yticklabels(genre_ratings.index)
axes[0,2].set_title('Top Rated Genres', fontweight='bold')
axes[0,2].set_xlabel('Average Rating')

# 4. Votes vs Rating
axes[1,0].scatter(df['Votes'], df['Rating'], alpha=0.5, s=10)
axes[1,0].set_xscale('log')
axes[1,0].set_xlabel('Votes (log scale)')
axes[1,0].set_ylabel('Rating')
axes[1,0].set_title('Votes vs Rating', fontweight='bold')

# 5. Duration vs Rating
axes[1,1].scatter(df['Duration'], df['Rating'], alpha=0.5, s=10, color='purple')
axes[1,1].set_xlabel('Duration (minutes)')
axes[1,1].set_ylabel('Rating')
axes[1,1].set_title('Duration vs Rating', fontweight='bold')

# 6. Year Trend
yearly_avg = df.groupby('Year')['Rating'].mean()
axes[1,2].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, color='orange')
axes[1,2].set_xlabel('Year')
axes[1,2].set_ylabel('Average Rating')
axes[1,2].set_title('Rating Trend Over Years', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{figures_path}/visualizations.png', dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print(f"✓ Visualizations saved to {figures_path}/visualizations.png")

# ============================================
# 5. TRAIN-TEST SPLIT
# ============================================
print("\n[5/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Training: {len(X_train)} movies")
print(f"✓ Testing: {len(X_test)} movies")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. TRAIN MODELS
# ============================================
print("\n[6/8] Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"  Training {name}...", end=" ", flush=True)
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test_scaled)
    
    results[name] = {
        'R² Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    print(f"✓ R² = {results[name]['R² Score']:.4f}")

# ============================================
# 7. MODEL COMPARISON
# ============================================
print("\n[7/8] Model Performance Summary:")
print("="*60)
results_df = pd.DataFrame(results).T
print(results_df.round(4))
print("="*60)

# Find best model
best_model_name = results_df['R² Score'].idxmax()
best_model = trained_models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {results_df.loc[best_model_name, 'R² Score']:.4f}")
print(f"   MAE: {results_df.loc[best_model_name, 'MAE']:.4f}")
print(f"   RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}")

# ============================================
# 8. SAVE MODEL
# ============================================
print("\n[8/8] Saving model...")

joblib.dump(best_model, f'{models_path}/best_model.pkl')
joblib.dump(scaler, f'{models_path}/scaler.pkl')
joblib.dump(le, f'{models_path}/genre_encoder.pkl')

print(f"✓ best_model.pkl saved to {models_path}/")
print(f"✓ scaler.pkl saved to {models_path}/")
print(f"✓ genre_encoder.pkl saved to {models_path}/")

# ============================================
# TEST PREDICTIONS
# ============================================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

def predict_movie_rating(duration, votes, year, genre):
    """Predict movie rating"""
    movie_age = 2024 - year
    
    # Encode genre
    try:
        genre_encoded = le.transform([genre])[0]
    except:
        genre_encoded = le.transform(['Unknown'])[0]
    
    # Create features
    features = np.array([[duration, votes, year, movie_age, genre_encoded]])
    features_scaled = scaler.transform(features)
    
    # Predict
    rating = best_model.predict(features_scaled)[0]
    return round(max(1.0, min(10.0, rating)), 2)

# Test cases
test_movies = [
    ("3 Idiots", 170, 400000, 2009, "Comedy"),
    ("RRR", 180, 500000, 2022, "Action"),
    ("Dangal", 161, 300000, 2016, "Biography"),
    ("Drishyam", 163, 100000, 2015, "Thriller"),
    ("Zindagi Na Milegi Dobara", 155, 80000, 2011, "Drama"),
    ("Andhadhun", 139, 60000, 2018, "Thriller"),
    ("Kabir Singh", 172, 100000, 2019, "Romance"),
]

print("\nMovie Predictions:")
print("-"*75)
for name, dur, votes, year, genre in test_movies:
    pred = predict_movie_rating(dur, votes, year, genre)
    print(f"{name:30s} | {dur:3d}min | {votes:>8,} votes | Predicted: {pred}/10")
print("-"*75)

# ============================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================
if hasattr(best_model, 'feature_importances_'):
    print("\n📊 Feature Importance:")
    print("-"*40)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for i, row in feature_importance.iterrows():
        print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
print(f"""
✅ Dataset: {len(df)} Indian movies analyzed
✅ Features: {X.shape[1]} features created
✅ Models: {len(models)} models trained
✅ Best Model: {best_model_name}
✅ Performance: R² = {results_df.loc[best_model_name, 'R² Score']:.4f}
✅ MAE = {results_df.loc[best_model_name, 'MAE']:.4f}
✅ RMSE = {results_df.loc[best_model_name, 'RMSE']:.4f}

📁 Files Created:
   - {models_path}/best_model.pkl
   - {models_path}/scaler.pkl  
   - {models_path}/genre_encoder.pkl
   - {figures_path}/visualizations.png

🚀 Next: Run Streamlit Web App
   cd D:\Projects\CineRate
   streamlit run app/streamlit_app.py
""")
print("="*60)