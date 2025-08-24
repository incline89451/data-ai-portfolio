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
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
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
    X = df[num_cols].copy()
    y = df["label"].copy()

    out = pd.concat([df[["yearID"]], X, y], axis=1)
    return out, num_cols

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

    # Clean features
    X = df[feat_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    y = df["label"].values.ravel()   # ensure y is strictly 1D
    X = X.values


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
    plot_confusion_matrix(cm, class_names=("No DivWin","DivWin"))

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
    st.caption("Loads Teams.parquet from ./data in your GitHub repo")
    task_team_divwin()

    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("""
    - Place Teams.parquet under `./data/` in your GitHub repo (Streamlit Cloud clones the repo).
    - Target: `DivWin` (1 if team won division, else 0).
    - Features: numeric stats from Teams table excluding IDs and non-numeric columns.
    - All missing or infinite values are replaced with 0 for model training.
    """)

if __name__ == "__main__":
    main()

