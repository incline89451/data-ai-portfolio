# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Data
# -----------------------------
DATA_PATH = "data/People.parquet"

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

df = load_data()

st.title("⚾ Lahman Baseball Simple Prediction")
st.write("Using **People.parquet** with scikit-learn + Streamlit")

# -----------------------------
# Simple Features & Target
# -----------------------------
# We'll predict batting hand (bats) using height/weight as features
df = df[['height', 'weight', 'bats']].dropna()
df = df[df['bats'].isin(['L', 'R'])]  # keep only left/right batters

X = df[['height', 'weight']]
y = df['bats']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: **{acc:.2f}**")

# -----------------------------
# User Input for Prediction
# -----------------------------
st.subheader("Try Your Own Prediction")

height = st.slider("Player Height (inches)", int(df['height'].min()), int(df['height'].max()), 72)
weight = st.slider("Player Weight (lbs)", int(df['weight'].min()), int(df['weight'].max()), 180)

if st.button("Predict Batting Hand"):
    pred = model.predict([[height, weight]])[0]
    st.success(f"Predicted Batting Hand: **{pred}**")
