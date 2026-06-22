import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time  # Add this for delays

# Page config
st.set_page_config(
    page_title="CineRate - Movie Rating Predictor",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 CineRate - Movie Rating Prediction System")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        genre_encoder = joblib.load('models/genre_encoder.pkl')
        return model, scaler, genre_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training script first: python notebooks/cinerate_no_graphs.py")
        return None, None, None

model, scaler, genre_encoder = load_models()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎭 Movie Details")
    
    movie_name = st.text_input("Movie Name", placeholder="Enter movie name")
    
    genre = st.selectbox(
        "Genre",
        ["Action", "Comedy", "Drama", "Thriller", "Romance", "Horror", 
         "Sci-Fi", "Biography", "Adventure", "Crime", "Mystery", "Family"]
    )
    
    director = st.text_input("Director", placeholder="Enter director name")

with col2:
    st.subheader("⚙️ Technical Details")
    
    duration = st.slider("Duration (minutes)", 60, 240, 150)
    votes = st.number_input("Number of Votes", min_value=0, max_value=1000000, value=50000)
    year = st.slider("Release Year", 1950, 2024, 2020)

# Prediction function with progress
def predict_rating_with_progress(duration, votes, year, genre, progress_bar, status_text):
    try:
        # Step 1: Calculate movie age (20%)
        status_text.text("📊 Calculating movie features...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        movie_age = 2024 - year
        
        # Step 2: Encode genre (40%)
        status_text.text("🎭 Encoding genre...")
        progress_bar.progress(40)
        time.sleep(0.3)
        
        try:
            genre_encoded = genre_encoder.transform([genre])[0]
        except:
            genre_encoded = 0
        
        # Step 3: Create feature array (60%)
        status_text.text("🔧 Preparing feature matrix...")
        progress_bar.progress(60)
        time.sleep(0.3)
        
        features = np.array([[duration, votes, year, movie_age, genre_encoded]])
        
        # Step 4: Scale features (80%)
        status_text.text("⚡ Scaling features...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        features_scaled = scaler.transform(features)
        
        # Step 5: Make prediction (100%)
        status_text.text("🎯 Making prediction...")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        rating = model.predict(features_scaled)[0]
        
        return round(max(1.0, min(10.0, rating)), 1)
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 6.5

# Predict button
if st.button("🎯 Predict Rating", type="primary", use_container_width=True):
    if model is not None:
        # Create placeholders for progress bar
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize progress bar
        progress_bar = progress_placeholder.progress(0)
        status_text = status_placeholder.empty()
        status_text.text("⏳ Starting prediction process...")
        progress_bar.progress(5)
        time.sleep(0.3)
        
        # Make prediction with progress
        rating = predict_rating_with_progress(duration, votes, year, genre, progress_bar, status_text)
        
        # Clear progress indicators
        time.sleep(0.5)
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        # Create columns for result display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px;">
                <h2 style="color: white;">Predicted IMDb Rating</h2>
                <p style="font-size: 5rem; font-weight: bold; color: #ffd700; margin: 0;">{rating}/10</p>
                <p style="color: white; margin-top: 1rem;">
                    Based on {duration} minutes, {votes:,} votes, released in {year}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rating interpretation with progress bar
        st.markdown("### 💡 Rating Interpretation")
        
        if rating >= 8:
            st.success("🌟 **Excellent!** This movie has potential to be a classic!")
            st.progress(1.0, text="Rating Quality: Excellent")
        elif rating >= 7:
            st.info("👍 **Good!** This movie should perform well!")
            st.progress(0.8, text="Rating Quality: Good")
        elif rating >= 6:
            st.warning("📊 **Average!** This movie might get mixed reviews!")
            st.progress(0.6, text="Rating Quality: Average")
        else:
            st.error("⚠️ **Below Average!** This movie might struggle!")
            st.progress(0.3, text="Rating Quality: Below Average")
        
        # Additional insights with progress bars
        st.markdown("### 📈 Key Factors Analysis")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("**Duration Impact**")
            if duration < 120:
                st.success("✅ Positive (Under 2 hours)")
                st.progress(0.8, text="Duration Score")
            elif duration > 180:
                st.warning("⚠️ Negative (Very long)")
                st.progress(0.3, text="Duration Score")
            else:
                st.info("➖ Neutral (Standard length)")
                st.progress(0.6, text="Duration Score")
        
        with insight_col2:
            st.markdown("**Audience Reach**")
            if votes > 100000:
                st.success("✅ High (Popular movie)")
                st.progress(0.9, text="Popularity Score")
            elif votes < 1000:
                st.warning("⚠️ Low (Limited exposure)")
                st.progress(0.2, text="Popularity Score")
            else:
                st.info("➖ Moderate (Growing interest)")
                st.progress(0.5, text="Popularity Score")
        
        with insight_col3:
            st.markdown("**Recency**")
            if year >= 2020:
                st.success("✅ Recent (Modern audience)")
                st.progress(0.8, text="Recency Score")
            else:
                st.info("➖ Classic (Tested over time)")
                st.progress(0.6, text="Recency Score")
                
    else:
        st.error("⚠️ Model not loaded. Please run the training script first.")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 About")
    st.markdown("""
    CineRate predicts IMDb ratings using Machine Learning.
    
    **Features used:**
    - Duration
    - Number of votes
    - Release year
    - Genre
    
    **Best Model:** Random Forest
    **R² Score:** 0.71
    **MAE:** 0.52
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>CineRate - Movie Rating Prediction System | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)