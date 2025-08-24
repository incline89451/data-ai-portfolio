#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app for modeling with the Lahman Baseball Database parquet schemas under ./data/
Tasks:
1) Hall of Fame Prediction (Classification)
2) Salary Estimation (Regression)
3) Team Division Win Prediction (Classification)
4) Player Clustering (Unsupervised - KMeans)
5) Player PCA Visualization (Unsupervised - PCA)
"""
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def find_parquet_file(preferred_names):
    """
    Return a path under DATA_DIR for the first existing filename in preferred_names.
    Also try case-insensitive matches by scanning directory once if needed.
    """
    # Try preferred names directly
    for name in preferred_names:
        p = DATA_DIR / f"{name}.parquet"
        if p.exists():
            return p

    # Case-insensitive scan
    if DATA_DIR.exists():
        lower_map = {f.name.lower(): f for f in DATA_DIR.glob("*.parquet")}
        for name in preferred_names:
            key = f"{name}.parquet".lower()
            if key in lower_map:
                return lower_map[key]
    return None

@st.cache_data(show_spinner=False)
def load_parquet_table(preferred_names, columns=None):
    path = find_parquet_file(preferred_names)
    if path is None:
        st.warning(f"Could not find any of {preferred_names} as .parquet under '{DATA_DIR}/'.")
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=columns)
    return df

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def plot_confusion_matrix(cm, class_names=("Negative","Positive")):
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

def simple_scatter(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)

# ----------------------------
# Feature builders
# ----------------------------
@st.cache_data(show_spinner=True)
def build_career_stats():
    batting = load_parquet_table(["Batting"])
    pitching = load_parquet_table(["Pitching"])
    fielding = load_parquet_table(["Fielding"])

    if batting.empty and pitching.empty and fielding.empty:
        return pd.DataFrame()

    # Sum per player across career
    dfs = []
    if not batting.empty:
        b = batting.copy()
        num_cols = [c for c in b.columns if c not in ("playerID","teamID","lgID","yearID","stint")]
        b = ensure_numeric(b, num_cols)
        b_agg = b.groupby("playerID")[num_cols].sum(min_count=1).add_prefix("bat_")
        dfs.append(b_agg)

    if not pitching.empty:
        p = pitching.copy()
        num_cols = [c for c in p.columns if c not in ("playerID","teamID","lgID","yearID","stint")]
        p = ensure_numeric(p, num_cols)
        p_agg = p.groupby("playerID")[num_cols].sum(min_count=1).add_prefix("pit_")
        dfs.append(p_agg)

    if not fielding.empty:
        f = fielding.copy()
        num_cols = [c for c in f.columns if c not in ("playerID","teamID","lgID","yearID","stint","Pos")]
        f = ensure_numeric(f, num_cols)
        f_agg = f.groupby("playerID")[num_cols].sum(min_count=1).add_prefix("fld_")
        dfs.append(f_agg)

    if not dfs:
        return pd.DataFrame()

    career = pd.concat(dfs, axis=1, join="outer").reset_index()
    return career

@st.cache_data(show_spinner=True)
def build_hof_dataset():
    hof = load_parquet_table(["HallOfFame","HallofFame"])
    career = build_career_stats()

    if hof.empty or career.empty:
        return pd.DataFrame(), None

    # Label: player inducted at least once (Y) in any category containing 'Player' or overall Y
    hof_simple = hof.copy()
    # Normalize columns
    for col in ["inducted","category"]:
        if col not in hof_simple.columns:
            hof_simple[col] = np.nan
    # per player label
    hof_label = (hof_simple.groupby("playerID")["inducted"]
                 .apply(lambda s: int((s.astype(str) == "Y").any()))).reset_index()
    hof_label = hof_label.rename(columns={"inducted":"label"})

    data = career.merge(hof_label, on="playerID", how="left")
    data["label"] = data["label"].fillna(0).astype(int)

    # Basic feature selection: keep numeric columns only
    feature_cols = [c for c in data.columns if c != "playerID" and pd.api.types.is_numeric_dtype(data[c])]
    X = data[feature_cols].fillna(0)
    y = data["label"]

    return (pd.concat([data[["playerID","label"]], X], axis=1)), feature_cols

@st.cache_data(show_spinner=True)
def build_salary_regression_dataset():
    salaries = load_parquet_table(["Salaries"])
    batting = load_parquet_table(["Batting"])
    pitching = load_parquet_table(["Pitching"])

    if salaries.empty:
        return pd.DataFrame(), []

    # Prior-year stats as features
    feats = []
    if not batting.empty:
        b = batting.copy()
        b_num = [c for c in b.columns if c not in ("playerID","teamID","lgID","yearID","stint")]
        b = ensure_numeric(b, b_num)
        b["yearID"] = pd.to_numeric(b["yearID"], errors="coerce")
        b_prev = b.copy()
        b_prev["yearID"] = b_prev["yearID"] + 1  # shift forward so prev-year aligns with salary year
        b_prev = b_prev.groupby(["playerID","yearID"])[b_num].sum(min_count=1).add_prefix("bat_prev_").reset_index()
        feats.append(b_prev)

    if not pitching.empty:
        p = pitching.copy()
        p_num = [c for c in p.columns if c not in ("playerID","teamID","lgID","yearID","stint")]
        p = ensure_numeric(p, p_num)
        p["yearID"] = pd.to_numeric(p["yearID"], errors="coerce")
        p_prev = p.copy()
        p_prev["yearID"] = p_prev["yearID"] + 1
        p_prev = p_prev.groupby(["playerID","yearID"])[p_num].sum(min_count=1).add_prefix("pit_prev_").reset_index()
        feats.append(p_prev)

    base = salaries.copy()
    base["yearID"] = pd.to_numeric(base["yearID"], errors="coerce")
    base = base.dropna(subset=["yearID","salary","playerID"])
    base["salary"] = pd.to_numeric(base["salary"], errors="coerce")
    base = base.dropna(subset=["salary"])

    df = base[["playerID","yearID","salary"]].copy()
    for f in feats:
        df = df.merge(f, on=["playerID","yearID"], how="left")

    # Remove rows with no features
    feat_cols = [c for c in df.columns if c not in ("playerID","yearID","salary")]
    if not feat_cols:
        return pd.DataFrame(), []

    df = df.dropna(subset=feat_cols, how="all")
    df[feat_cols] = df[feat_cols].fillna(0)

    return df, feat_cols

@st.cache_data(show_spinner=True)
def build_team_divwin_dataset():
    teams = load_parquet_table(["Teams"])
    if teams.empty:
        return pd.DataFrame(), []

    # Binary label: DivWin Y/N
    df = teams.copy()
    if "DivWin" not in df.columns:
        return pd.DataFrame(), []
    df["label"] = (df["DivWin"].astype(str) == "Y").astype(int)

    exclude = {"DivWin","WCWin","LgWin","WSWin","name","park","teamIDBR","teamIDlahman45","teamIDretro","teamID","lgID","franchID","divID"}
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num_cols].copy()
    y = df["label"].copy()

    out = pd.concat([df[["yearID"]], X, y], axis=1)
    return out, num_cols

@st.cache_data(show_spinner=True)
def build_player_feature_matrix():
    career = build_career_stats()
    if career.empty:
        return pd.DataFrame(), []
    # Rate features for batting
    X = career.copy()
    if "bat_AB" in X.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            X["bat_HR_rate"] = np.where(X["bat_AB"]>0, X.get("bat_HR", 0) / X["bat_AB"], 0)
            X["bat_BB_rate"] = np.where(X["bat_AB"]>0, X.get("bat_BB", 0) / X["bat_AB"], 0)
            X["bat_SO_rate"] = np.where(X["bat_AB"]>0, X.get("bat_SO", 0) / X["bat_AB"], 0)
            X["bat_SB_rate"] = np.where(X["bat_AB"]>0, X.get("bat_SB", 0) / X["bat_AB"], 0)
            X["bat_3B_rate"] = np.where(X["bat_AB"]>0, X.get("bat_3B", 0) / X["bat_AB"], 0)
            X["bat_2B_rate"] = np.where(X["bat_AB"]>0, X.get("bat_2B", 0) / X["bat_AB"], 0)
            # pseudo OPS proxy using HR, BB, H
            X["bat_H_rate"]  = np.where(X["bat_AB"]>0, X.get("bat_H", 0) / X["bat_AB"], 0)

    # Pitching rate features
    if "pit_IPOuts" in X.columns:
        IP = X["pit_IPOuts"] / 3.0
        with np.errstate(divide='ignore', invalid='ignore'):
            X["pit_SO_per9"] = np.where(IP>0, X.get("pit_SO", 0)/IP*9.0, 0)
            X["pit_BB_per9"] = np.where(IP>0, X.get("pit_BB", 0)/IP*9.0, 0)
            X["pit_HR_per9"] = np.where(IP>0, X.get("pit_HR", 0)/IP*9.0, 0)
            X["pit_ERA"]     = X.get("pit_ERA", 0)

    # Keep numeric features
    feat_cols = [c for c in X.columns if c != "playerID" and pd.api.types.is_numeric_dtype(X[c])]
    X = X[["playerID"] + feat_cols].fillna(0)
    return X, feat_cols

# ----------------------------
# Task implementations
# ----------------------------
def task_hof_classification():
    st.subheader("Hall of Fame Prediction (Classification)")
    data, feature_cols = build_hof_dataset()
    if data.empty:
        st.error("Required data not found. Make sure HallOfFame/HallofFame, Batting/Pitching/Fielding parquet files exist in ./data/.")
        return

    model_choice = st.selectbox("Choose model", ["Logistic Regression","Random Forest"])
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    X = data[feature_cols].values
    y = data["label"].values

    # Scale for logistic regression; RF doesn't need it
    if model_choice == "Logistic Regression":
        X, scaler = standardize_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 4))
    # Probabilities for AUC if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        st.write("**ROC AUC:**", round(roc_auc_score(y_test, y_prob), 4))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names=("Not Inducted","Inducted"))

    # Top features
    st.markdown("**Top Features**")
    if model_choice == "Random Forest":
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            idx = np.argsort(importances)[-20:][::-1]
            top_feats = [(feature_cols[i], float(importances[i])) for i in idx]
            st.dataframe(pd.DataFrame(top_feats, columns=["feature","importance"]))
    else:
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            idx = np.argsort(np.abs(coefs))[-20:][::-1]
            top_feats = [(feature_cols[i], float(coefs[i])) for i in idx]
            st.dataframe(pd.DataFrame(top_feats, columns=["feature","coefficient"]))


def task_salary_regression():
    st.subheader("Salary Estimation (Regression)")
    df, feat_cols = build_salary_regression_dataset()
    if df.empty:
        st.error("Required data not found. Make sure Salaries and Batting/Pitching parquet files exist in ./data/.")
        return

    model_choice = st.selectbox("Choose model", ["Linear Regression","Random Forest Regressor"])
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    log_target = st.checkbox("Log-transform salary", value=True)

    X = df[feat_cols].values
    y = df["salary"].values
    if log_target:
        y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    if log_target:
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
        r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
    else:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    st.write("**MAE:**", round(mae, 2))
    st.write("**R²:**", round(r2, 4))

    # Plot predicted vs actual
    if log_target:
        actual = np.expm1(y_test)
        pred = np.expm1(y_pred)
    else:
        actual = y_test
        pred = y_pred
    simple_scatter(actual, pred, "Actual Salary", "Predicted Salary", "Predicted vs Actual Salary")

def task_team_divwin():
    st.subheader("Team Division Win Prediction (Classification)")
    df, feat_cols = build_team_divwin_dataset()
    if df.empty:
        st.error("Required data not found. Make sure Teams.parquet exists in ./data/.")
        return

    model_choice = st.selectbox("Choose model", ["Gradient Boosting","Random Forest"])
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    X = df[feat_cols].values
    y = df["label"].values

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

def task_player_clustering():
    st.subheader("Player Clustering (KMeans)")
    X, feat_cols = build_player_feature_matrix()
    if X.empty:
        st.error("Required data not found. Make sure Batting/Pitching/Fielding parquet files exist in ./data/.")
        return

    k = st.slider("Number of clusters (k)", min_value=3, max_value=12, value=6, step=1)
    use_pca = st.checkbox("Reduce to 2D with PCA for visualization", value=True)

    feats = X[feat_cols].values
    feats_scaled, scaler = standardize_features(feats)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(feats_scaled)

    if use_pca:
        pca = PCA(n_components=2, random_state=42)
        pts = pca.fit_transform(feats_scaled)
        # Plot clusters
        fig, ax = plt.subplots()
        ax.scatter(pts[:,0], pts[:,1], s=6, c=labels)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("KMeans Clusters (PCA-reduced)")
        st.pyplot(fig)

    # Show sample of players per cluster
    out = X[["playerID"]].copy()
    out["cluster"] = labels
    st.dataframe(out.groupby("cluster").head(10))

def task_player_pca():
    st.subheader("Player PCA Visualization")
    X, feat_cols = build_player_feature_matrix()
    if X.empty:
        st.error("Required data not found. Make sure Batting/Pitching/Fielding parquet files exist in ./data/.")
        return

    n_comp = st.slider("Number of components", min_value=2, max_value=10, value=2, step=1)
    feats = X[feat_cols].values
    feats_scaled, scaler = standardize_features(feats)

    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(feats_scaled)

    # 2D plot for first two PCs
    if n_comp >= 2:
        fig, ax = plt.subplots()
        ax.scatter(pcs[:,0], pcs[:,1], s=6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Players in PCA space (PC1 vs PC2)")
        st.pyplot(fig)

    st.write("**Explained variance ratio (first 10):**", np.round(pca.explained_variance_ratio_, 4))

# ----------------------------
# Main app
# ----------------------------
def main():
    st.title("Lahman Baseball Models (scikit-learn)")
    st.caption("Loads parquet files from ./data in your GitHub repo")

    # Quick file presence check
    expected = ["People","Batting","Pitching","Fielding","Teams","HallOfFame","HallofFame","Salaries"]
    present = []
    if DATA_DIR.exists():
        present = [p.name for p in DATA_DIR.glob("*.parquet")]
    st.write("**Detected parquet files in ./data/**:", present if present else "None found")

    task = st.selectbox(
        "Choose a task",
        [
            "Hall of Fame Prediction (Classification)",
            "Salary Estimation (Regression)",
            "Team Division Win Prediction (Classification)",
            "Player Clustering (KMeans)",
            "Player PCA Visualization (PCA)",
        ],
    )

    if task.startswith("Hall of Fame"):
        task_hof_classification()
    elif task.startswith("Salary"):
        task_salary_regression()
    elif task.startswith("Team Division"):
        task_team_divwin()
    elif task.startswith("Player Clustering"):
        task_player_clustering()
    else:
        task_player_pca()

    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("""
    - Place parquet files under `./data/` in your GitHub repo (Streamlit Cloud clones the repo).
    - Hall of Fame label is derived as 1 if a player has any 'Y' in `HallOfFame.inducted` across ballots.
    - Salary regression uses prior-year Batting/Pitching aggregates aligned to salary year.
    - All plots use matplotlib.
    """)

if __name__ == "__main__":
    main()
