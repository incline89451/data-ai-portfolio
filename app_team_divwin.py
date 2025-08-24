#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app: Team Division Win Prediction (Classification)
Using the Lahman Baseball Database Teams.parquet
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

DATA_DIR = Path("data")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def load_teams_table():
    path = DATA_DIR / "Teams.parquet"
    if not path.exists():
        st.error("Teams.parquet not found in ./data/. Please add it to your repo.")
        return pd.DataFrame()
    return pd.read_parquet(path)

def plot_confusion_matrix(cm, class_names=("No DivWin","DivWin")):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use a lighter colormap and set vmin/vmax for better contrast
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.6)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    
    # Add text with better visibility
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Use white text on dark cells, black text on light cells
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'), 
                   ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')
    st.pyplot(fig)

# ----------------------------
# Data builder
# ----------------------------
@st.cache_data(show_spinner=True)
def build_team_divwin_dataset():
    teams = load_teams_table()
    if teams.empty:
        return pd.DataFrame(), []

    df = teams.copy()
    if "DivWin" not in df.columns:
        return pd.DataFrame(), []

    df["label"] = (df["DivWin"].astype(str) == "Y").astype(int)

    exclude = {"DivWin","WCWin","LgWin","WSWin","name","park",
               "teamIDBR","teamIDlahman45","teamIDretro",
               "teamID","lgID","franchID","divID"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    
    # Return the original df with the label column added, not a concatenated version
    return df, num_cols

# ----------------------------
# Task implementation
# ----------------------------
def task_team_divwin():
    st.subheader("Team Division Win Prediction (Classification)")
    df, feat_cols = build_team_divwin_dataset()
    if df.empty:
        st.error("Required data not found. Make sure Teams.parquet exists in ./data/.")
        return

    model_choice = st.selectbox("Choose model", ["Gradient Boosting","Random Forest"])
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    # Work directly with the feature columns and label from df
    # Don't use the concatenated output from build_team_divwin_dataset
    feature_cols = [c for c in feat_cols if c in df.columns]
    
    # Clean features and labels together to maintain alignment
    X_df = df[feature_cols].copy()
    y_series = df["label"].copy()
    
    # Apply cleaning to features
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(0)
    
    # Ensure X and y have the same index
    common_idx = X_df.index.intersection(y_series.index)
    X_clean = X_df.loc[common_idx]
    y_clean = y_series.loc[common_idx]
    
    # Convert to numpy arrays
    X = X_clean.values
    y = y_clean.values.ravel()   # ensure y is strictly 1D
    
    # Add debugging info
    st.write(f"Data shape: X={X.shape}, y={y.shape}")
    st.write(f"Class distribution: {np.bincount(y)}")

    if len(X) != len(y):
        st.error(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples")
        return
    
    if len(np.unique(y)) < 2:
        st.error("Need at least 2 classes for classification")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 4))
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        st.write("**ROC AUC:**", round(roc_auc_score(y_test, y_prob), 4))

    cm = confusion_matrix(y_test, y_pred)
    
    # Add explanation before showing the confusion matrix
    st.markdown("**Confusion Matrix**")
    st.markdown("""
    This matrix shows how well our model predicted division winners:
    - **Top-left**: Teams we correctly predicted would NOT win their division
    - **Top-right**: Teams we incorrectly predicted would win (False Alarms)
    - **Bottom-left**: Teams we missed that actually won their division (Missed Winners)
    - **Bottom-right**: Teams we correctly predicted would win their division
    
    **Good performance** = Large numbers on the diagonal (top-left and bottom-right)
    """)
    
    plot_confusion_matrix(cm, class_names=("No DivWin","DivWin"))
    
    # Add interpretation after the matrix
    tn, fp, fn, tp = cm.ravel()
    st.markdown("**What these numbers mean:**")
    st.markdown(f"""
    - ✅ **Correctly predicted {tn + tp} teams** out of {len(y_test)} total
    - ❌ **Missed {fn} division winners** (predicted they wouldn't win, but they did)
    - ⚠️ **False alarms: {fp} teams** (predicted they'd win, but they didn't)
    - 🎯 **Found {tp} actual division winners** correctly
    """)
    
    if tp + fn > 0:  # Avoid division by zero
        recall = tp / (tp + fn)
        st.markdown(f"📊 **Recall**: We found {recall:.1%} of all actual division winners")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        st.markdown(f"🎯 **Precision**: {precision:.1%} of our 'division winner' predictions were correct")

    # Feature importance table
    st.markdown("**Top Features**")
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        idx = np.argsort(importances)[-20:][::-1]
        top_feats = [(feat_cols[i], float(importances[i])) for i in idx]
        st.dataframe(pd.DataFrame(top_feats, columns=["feature","importance"]))

# ----------------------------
# Main app
# ----------------------------
def main():
    st.title("Team Division Win Prediction (scikit-learn)")
    st.caption("Loads Teams.parquet from ./data in the GitHub repo")
    task_team_divwin()

    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("""
    - Target: `DivWin` (1 if team won division, else 0).
    - Features: numeric stats from Teams table excluding IDs and non-numeric columns.
    - All missing or infinite values are replaced with 0 for model training.
    """)

if __name__ == "__main__":
    main()
