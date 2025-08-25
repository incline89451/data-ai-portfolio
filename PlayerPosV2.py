
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import io

st.title("Baseball Player Position Classifier")

# Load data
@st.cache_data
def load_data():
    people = pd.read_parquet("data/People.parquet")
    batting = pd.read_parquet("data/Batting.parquet")
    pitching = pd.read_parquet("data/Pitching.parquet")
    hof = pd.read_parquet("data/HallOfFame.parquet")
    return people, batting, pitching, hof

people, batting, pitching, hof = load_data()

# --- Feature engineering (simplified example) ---
def derive_features(people, batting, pitching):
    bat_agg = batting.groupby("playerID").agg({"HR":"sum","RBI":"sum","AB":"sum"}).reset_index()
    pit_agg = pitching.groupby("playerID").agg({"SO":"sum","ERA":"mean"}).reset_index()
    feats = people.merge(bat_agg, on="playerID", how="left").merge(pit_agg, on="playerID", how="left")
    feats = feats.fillna(0)
    return feats

features = derive_features(people, batting, pitching)

# Create label (dummy example: position as label)
labels = people[["playerID","primaryPos"]] if "primaryPos" in people.columns else None

if labels is not None:
    data = features.merge(labels, on="playerID")
    X = data.drop(columns=["playerID","primaryPos"])
    y = data["primaryPos"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Overall Accuracy: {accuracy:.3f}")

    # --- Diagnostics ---
    # Class distribution
    st.write("### Class Distribution (Training Data)")
    class_counts = y.value_counts(normalize=True).sort_index()
    st.bar_chart(class_counts)

    # Classification Report
    st.write("### Per-Class Performance (Test Data)")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Download trained model
    st.write("### Download trained model (optional)")
    buffer = io.BytesIO()
    joblib.dump(clf, buffer)
    buffer.seek(0)
    st.download_button("Download Model", buffer, file_name="player_position_model.pkl")

else:
    st.error("No label column 'primaryPos' found in People.parquet.")
