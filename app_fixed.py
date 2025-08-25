
# app.py
# Streamlit app: Player Position Optimization using scikit-learn on Lahman-like parquet data
import os
import sys
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ------------ Configuration ------------
#DATA_DIR = os.environ.get("DATA_PATH", "/data")
DATA_DIR = "data"

# Candidate parquet filenames per table (case-insensitive)
PARQUET_NAMES = {
    "people": ["People.parquet", "people.parquet", "PEOPLE.parquet"],
    "appearances": ["Appearances.parquet", "appearances.parquet"],
    "fielding": ["Fielding.parquet", "fielding.parquet"],
}

# Positions we will predict (standard MLB positions)
POS_COLUMNS = [
    ("G_p", "P"), ("G_c", "C"), ("G_1b", "1B"), ("G_2b", "2B"),
    ("G_3b", "3B"), ("G_ss", "SS"), ("G_lf", "LF"),
    ("G_cf", "CF"), ("G_rf", "RF"),
]

MIN_GAMES_FOR_LABEL = 20  # minimum total games across tracked positions to assign a primary position label
RANDOM_STATE = 42


# ------------ Helpers ------------
def find_file(data_dir: str, candidates: List[str]) -> str:
    """Return first matching path in directory by case-insensitive name, else raise FileNotFoundError."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    existing = {f.lower(): os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".parquet")}
    for name in candidates:
        p = existing.get(name.lower())
        if p:
            return p
    # Fallback: fuzzy contains (e.g., LahmanPeople-2023.parquet -> contains 'people')
    for name in candidates:
        key = name.lower().replace(".parquet", "")
        for k, v in existing.items():
            if key in k:
                return v
    raise FileNotFoundError(f"None of {candidates} found in {data_dir}. Parquet files seen: {sorted(existing.keys())}")


@st.cache_data(show_spinner=False)
def load_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    people_path = find_file(data_dir, PARQUET_NAMES["people"])
    app_path = find_file(data_dir, PARQUET_NAMES["appearances"])
    fld_path = find_file(data_dir, PARQUET_NAMES["fielding"])

    people = pd.read_parquet(people_path, engine="pyarrow")
    appearances = pd.read_parquet(app_path, engine="pyarrow")
    fielding = pd.read_parquet(fld_path, engine="pyarrow")
    return {"people": people, "appearances": appearances, "fielding": fielding}


def build_dataset(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    people = tables["people"][["playerID", "height", "weight"]].copy()

    # Ensure numeric height/weight; drop missing
    for col in ["height", "weight"]:
        people[col] = pd.to_numeric(people[col], errors="coerce")

    # Aggregate Appearances across seasons to get total games per position
    app = tables["appearances"].copy()
    # Keep only the columns we need
    keep_cols = ["playerID"] + [c for c, _ in POS_COLUMNS]
    app = app[keep_cols].copy()
    app_agg = app.groupby("playerID", as_index=False).sum(numeric_only=True)

    # Determine primary position label
    pos_counts = app_agg[[c for c, _ in POS_COLUMNS]].fillna(0)
    total_games = pos_counts.sum(axis=1)
    primary_idx = pos_counts.values.argmax(axis=1)
    labels = pd.Series([POS_COLUMNS[i][1] for i in primary_idx], index=app_agg.index, name="primary_pos")
    app_agg["total_games_tracked"] = total_games
    app_agg["primary_pos"] = labels

    # Keep only players with enough games
    app_eligible = app_agg[app_agg["total_games_tracked"] >= MIN_GAMES_FOR_LABEL].copy()

    # Fielding aggregates per player
    fld = tables["fielding"].copy()
    fld_numeric = fld[["playerID", "G", "GS", "InnOuts", "PO", "A", "E", "DP"]].copy()
    fld_numeric[["G", "GS", "InnOuts", "PO", "A", "E", "DP"]] = fld_numeric[["G", "GS", "InnOuts", "PO", "A", "E", "DP"]].apply(
        pd.to_numeric, errors="coerce"
    )
    fld_agg = (
        fld_numeric.groupby("playerID", as_index=False)
        .sum(numeric_only=True)
        .rename(columns={"G": "fld_G", "GS": "fld_GS", "InnOuts": "fld_InnOuts", "PO": "fld_PO", "A": "fld_A", "E": "fld_E", "DP": "fld_DP"})
    )
    # Derived fielding ratios
    fld_agg["fld_chances"] = fld_agg[["fld_PO", "fld_A", "fld_E"]].sum(axis=1)
    fld_agg["fld_fpct"] = (fld_agg["fld_PO"] + fld_agg["fld_A"]) / fld_agg["fld_chances"].replace(0, np.nan)
    fld_agg["fld_dp_per_g"] = fld_agg["fld_DP"] / fld_agg["fld_G"].replace(0, np.nan)

    # Merge features
    feats = (
        app_eligible.merge(people, on="playerID", how="left")
        .merge(fld_agg, on="playerID", how="left")
    )

    # Per-game normalized versions of position counts (share of games by position)
    for col, short in POS_COLUMNS:
        feats[f"share_{short}"] = feats[col] / feats["total_games_tracked"]

    # Selected feature columns
    feature_cols = (
        ["height", "weight", "fld_G", "fld_GS", "fld_InnOuts", "fld_PO", "fld_A", "fld_E", "fld_DP", "fld_chances", "fld_fpct", "fld_dp_per_g"]
        + [f"share_{short}" for _, short in POS_COLUMNS]
    )

    # Clean
    X = feats[feature_cols].copy()
    y = feats["primary_pos"].copy()
    # Replace inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Keep for later lookup
    meta_cols = ["playerID", "primary_pos", "total_games_tracked"]
    meta = feats[meta_cols].copy()

    return X, y, meta


def train_model(X: pd.DataFrame, y: pd.Series, model_name: str):
    if model_name == "SVM (rbf)":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ])
    elif model_name == "Decision Tree":
        pipe = Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE))
        ])
    else:
        raise ValueError("Unknown model: " + model_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model = pipe.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    report = classification_report(y_test, y_pred, labels=sorted(y.unique()), output_dict=False)
    return model, acc, cm, report, X_test.index, y_test, y_pred


def plot_cm(cm: np.ndarray, class_labels: List[str]):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig


# ------------ Streamlit UI ------------
st.set_page_config(page_title="Player Position Optimization", layout="wide")
st.title("⚾ Player Position Optimization")
st.caption("Classify players into their most effective positions using fielding stats and physical attributes (Lahman schemas).")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Data directory", value=DATA_DIR, help="Directory that holds the parquet files (People, Appearances, Fielding).")
    model_choice = st.selectbox("Model", ["SVM (rbf)", "Decision Tree"], help="SVM for accuracy, Decision Tree for interpretability.")
    min_games = st.slider("Minimum total games to define a primary position", 0, 200, MIN_GAMES_FOR_LABEL, 5)
    st.caption("Tip: Increase the threshold for more reliable labels.")

# allow overriding global MIN_GAMES_FOR_LABEL
MIN_GAMES_FOR_LABEL = int(min_games)

status = st.empty()
try:
    status.info("Loading data…")
    tables = load_tables(data_dir)
    X, y, meta = build_dataset(tables)
    status.success(f"Loaded {len(X):,} players with labels across {y.nunique()} positions.")
except Exception as e:
    st.error(f"Failed to load/build dataset: {e}")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Train & Evaluate")
    model, acc, cm, report, test_idx, y_test, y_pred = train_model(X, y, model_choice)
    st.write(f"**Accuracy:** {acc:.3f}")
    fig = plot_cm(cm, sorted(y.unique()))
    st.pyplot(fig)

    if model_choice == "Decision Tree":
        st.subheader("Decision Tree (visual excerpt)")
        # Show a small tree to avoid huge plots
        try:
            tree_clf = model.named_steps.get("clf", None) or model  # if not in pipeline
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            plot_tree(tree_clf, max_depth=3, fontsize=8, feature_names=list(X.columns), class_names=sorted(y.unique()), filled=False)
            st.pyplot(fig2)
        except Exception as ex:
            st.info(f"Tree visualization unavailable: {ex}")

with col2:
    st.subheader("Predict for a Player")
    # Build a lookup of test players for quick demo
    test_meta = meta.loc[test_idx].copy()
    merged = test_meta.copy()
    merged["true"] = y_test.values
    merged["pred"] = y_pred
    merged["correct"] = (merged["true"] == merged["pred"])
    # Pick a player to preview
    if len(merged) > 0:
        player_row = st.selectbox("Choose a test-set playerID", merged["playerID"].tolist())
        row = merged[merged["playerID"] == player_row].iloc[0]
        st.metric("True position", row["true"])
        st.metric("Predicted position", row["pred"])
        st.write(f"Total games considered: {int(row['total_games_tracked'])}")
        st.write(f"Correct? {'✅' if row['correct'] else '❌'}")
    else:
        st.info("No players available in test set.")

st.divider()
st.subheader("Download trained model (optional)")
model_bytes = joblib.dumps(model)
st.download_button("Download model.pkl", data=model_bytes, file_name="position_model.pkl")

st.caption("Data schemas follow the Lahman Baseball Database (People, Appearances, Fielding tables).")
