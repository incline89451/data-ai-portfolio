import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# === Load Data ===
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("data/People.parquet")
    except Exception as e:
        st.error(f"❌ Could not load People.parquet: {e}")
        return None
    return df

df = load_data()

if df is None:
    st.stop()

st.title("⚾ Player Position Classifier")
st.write("Train a model to predict a player's position based on available data.")

# === Detect Label Column ===
possible_labels = ["primaryPos", "pos", "position", "playerPos"]
label_col = None
for col in possible_labels:
    if col in df.columns:
        label_col = col
        break

if not label_col:
    st.error(f"❌ No label column found. Expected one of {possible_labels}, but got: {list(df.columns)}")
    st.stop()
else:
    st.success(f"✅ Using **{label_col}** as the label column.")

# === Prepare Features & Labels ===
X = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=["number"]).fillna(0)
y = df[label_col]

if X.empty:
    st.error("❌ No numeric features found in dataset for training.")
    st.stop()

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predictions & Metrics ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Overall Accuracy:** {accuracy:.3f}")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# ROC AUC (only if binary classification)
if len(y.unique()) == 2:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"**ROC AUC Score:** {roc_auc:.3f}")

# === Model Download ===
st.subheader("💾 Download Trained Model")
joblib.dump(model, "rf_model.pkl")
with open("rf_model.pkl", "rb") as f:
    st.download_button("Download Model", f, file_name="rf_model.pkl")
