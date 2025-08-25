# Write a full, corrected Streamlit app that auto-detects labels from Appearances/Fielding
from textwrap import dedent
from pathlib import Path

app_code = dedent("""
# app.py
# Streamlit app: Player Position Optimization using scikit-learn on Lahman-like parquet data
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ------------ Configuration ------------
DATA_DIR = "data"

# Candidate parquet filenames per table (case-insensitive)
PARQUET_NAMES = {
    "people": ["People.parquet", "people.parquet", "PEOPLE.parquet"],
    "appearances": ["Appearances.parquet", "appearances.parquet"],
    "fielding": ["Fielding.parquet", "fielding.parquet"],
}

# Valid MLB positions (for labels)
VALID_POS = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]

# Mapping for Appearances columns (case-insensitive)
POS_COLUMNS = [
    ("g_p", "P"), ("g_c", "C"), ("g_1b", "1B"), ("g_2b", "2B"),
    ("g_3b", "3B"), ("g_ss", "SS"), ("g_lf", "LF"), ("g_cf", "CF"), ("g_rf", "RF"),
]

MIN_GAMES_FOR_LABEL_DEFAULT = 20  # default minimum games for creating a label
RANDOM_STATE = 42

# ------------ Helpers ------------
def find_file(data_dir: str, candidates: List[str]) -> str:
    \"\"\"Return first matching path in directory by case-insensitive name, else raise FileNotFoundError.\"\"\"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".parquet")]
    existing = {f.lower(): os.path.join(data_dir, f) for f in files}
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

def derive_primary_from_appearances(app: pd.DataFrame, min_games: int) -> pd.DataFrame:
    \"\"\"Derive primary position and total tracked games from Appearances (if G_* columns exist).\"\"\"
    if app is None or app.empty:
        return pd.DataFrame(columns=["playerID", "primary_pos", "total_games_tracked"])

    cols_lower = {c.lower(): c for c in app.columns}
    present = [(cols_lower[k], short) for k, short in POS_COLUMNS if k in cols_lower]
    if not present:
        return pd.DataFrame(columns=["playerID", "primary_pos", "total_games_tracked"])

    keep_cols = ["playerID"] + [orig for orig, _ in present]
    app2 = app[keep_cols].copy()
    # ensure numeric
    for orig, _ in present:
        app2[orig] = pd.to_numeric(app2[orig], errors="coerce").fillna(0)

    agg = app2.groupby("playerID", as_index=False).sum(numeric_only=True)

    # position counts matrix in a fixed order
    pos_counts = []
    short_labels = []
    for orig, short in present:
        pos_counts.append(agg[orig])
        short_labels.append(short)
    pos_mat = np.vstack(pos_counts).T  # shape (n_players, n_present_positions)
    total_games = pos_mat.sum(axis=1)

    # avoid empty rows
    total_games_series = pd.Series(total_games, index=agg.index, name="total_games_tracked")
    primary_idx = pos_mat.argmax(axis=1)
    labels = pd.Series([short_labels[i] for i in primary_idx], index=agg.index, name="primary_pos")

    out = agg[["playerID"]].copy()
    out["total_games_tracked"] = total_games_series
    out["primary_pos"] = labels
    out = out[out["total_games_tracked"] >= min_games].copy()
    return out

def derive_primary_from_fielding(fld: pd.DataFrame, min_games: int) -> pd.DataFrame:
    \"\"\"Fallback: derive primary position using Fielding table (uses POS + G).\"\"\"
    if fld is None or fld.empty:
        return pd.DataFrame(columns=["playerID", "primary_pos", "total_games_tracked"])

    cols_lower = {c.lower(): c for c in fld.columns}
    if "pos" not in cols_lower or "g" not in cols_lower:
        return pd.DataFrame(columns=["playerID", "primary_pos", "total_games_tracked"])

    pos_col = cols_lower["pos"]
    g_col = cols_lower["g"]
    fld2 = fld[["playerID", pos_col, g_col]].copy()
    fld2[g_col] = pd.to_numeric(fld2[g_col], errors="coerce").fillna(0)
    fld2 = fld2[fld2[pos_col].isin(VALID_POS)]

    # sum games by player & position
    gp = fld2.groupby(["playerID", pos_col], as_index=False)[g_col].sum()
    # total by player for tracked positions
    totals = gp.groupby("playerID", as_index=False)[g_col].sum().rename(columns={g_col: "total_games_tracked"})
    # choose primary as max G
    idx = gp.groupby("playerID")[g_col].idxmax()
    primary = gp.loc[idx, ["playerID", pos_col]].rename(columns={pos_col: "primary_pos"})
    out = primary.merge(totals, on="playerID", how="left")
    out = out[out["total_games_tracked"] >= min_games].copy()
    return out

def build_dataset(tables: Dict[str, pd.DataFrame], min_games: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    people = tables["people"][["playerID", "height", "weight"]].copy()

    # Ensure numeric height/weight; drop missing -> fill later
    for col in ["height", "weight"]:
        people[col] = pd.to_numeric(people[col], errors="coerce")

    # Derive labels (auto-detect): prefer Appearances, fallback to Fielding
    labels_from_app = derive_primary_from_appearances(tables["appearances"], min_games)
    if not labels_from_app.empty:
        labels_df = labels_from_app
        source = "Appearances"
    else:
        labels_df = derive_primary_from_fielding(tables["fielding"], min_games)
        source = "Fielding"

    if labels_df.empty:
        raise ValueError("Could not derive primary positions from Appearances or Fielding. "
                         "Ensure your parquet files contain either G_* columns in Appearances or POS/G in Fielding.")

    # Fielding aggregates per player for features
    fld = tables["fielding"].copy()
    # Pick common numeric columns if present
    num_cols = [c for c in ["G", "GS", "InnOuts", "PO", "A", "E", "DP"] if c in fld.columns]
    fld_numeric = fld[["playerID"] + num_cols].copy()
    for c in num_cols:
        fld_numeric[c] = pd.to_numeric(fld_numeric[c], errors="coerce")

    # aggregate
    fld_agg = (
        fld_numeric.groupby("playerID", as_index=False)
        .sum(numeric_only=True)
        .rename(columns={
            "G": "fld_G", "GS": "fld_GS", "InnOuts": "fld_InnOuts",
            "PO": "fld_PO", "A": "fld_A", "E": "fld_E", "DP": "fld_DP"
        })
    )
    # Derived fielding ratios (guard against missing columns)
    if set(["fld_PO", "fld_A", "fld_E"]).issubset(fld_agg.columns):
        fld_agg["fld_chances"] = fld_agg[["fld_PO", "fld_A", "fld_E"]].sum(axis=1)
        fld_agg["fld_fpct"] = (fld_agg["fld_PO"] + fld_agg["fld_A"]) / fld_agg["fld_chances"].replace(0, np.nan)
    else:
        fld_agg["fld_chances"] = np.nan
        fld_agg["fld_fpct"] = np.nan

    if "fld_DP" in fld_agg.columns and "fld_G" in fld_agg.columns:
        fld_agg["fld_dp_per_g"] = fld_agg["fld_DP"] / fld_agg["fld_G"].replace(0, np.nan)
    else:
        fld_agg["fld_dp_per_g"] = np.nan

    # Merge features
    feats = (
        labels_df.merge(people, on="playerID", how="left")
        .merge(fld_agg, on="playerID", how="left")
    )

    # Selected feature columns (numeric only; fill missing)
    candidate_features = ["height", "weight", "fld_G", "fld_GS", "fld_InnOuts", "fld_PO", "fld_A", "fld_E",
                          "fld_DP", "fld_chances", "fld_fpct", "fld_dp_per_g"]
    feature_cols = [c for c in candidate_features if c in feats.columns]
    X = feats[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = feats["primary_pos"].copy()
    meta = feats[["playerID", "primary_pos", "total_games_tracked"]].copy()

    return X, y, meta, source

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
    report_txt = classification_report(y_test, y_pred, labels=sorted(y.unique()), output_dict=False)
    report_dict = classification_report(y_test, y_pred, labels=sorted(y.unique()), output_dict=True)

    return model, acc, cm, report_txt, report_dict, X_test.index, y_test, y_pred

def plot_cm(cm: np.ndarray, class_labels: List[str]):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                annot_kws={"size": 12})
    ax.set_ylabel("Actual Position", fontsize=12)
    ax.set_xlabel("Predicted Position", fontsize=12)
    ax.set_title("Confusion Matrix: Actual vs Predicted Player Positions", fontsize=14)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig

# ------------ Streamlit UI ------------
st.set_page_config(page_title="Player Position Optimization", layout="wide")
st.title("⚾ Player Position Optimization")
st.caption("Classify players into their most effective positions using fielding stats and physical attributes (auto-detects labels from Appearances/Fielding).")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Data directory", value=DATA_DIR, help="Relative path that holds the parquet files.")
    model_choice = st.selectbox("Model", ["SVM (rbf)", "Decision Tree"], help="SVM for accuracy, Decision Tree for interpretability.")
    min_games = st.slider("Minimum total games to define a primary position", 0, 200, MIN_GAMES_FOR_LABEL_DEFAULT, 5)
    st.caption("Tip: Increase the threshold for more reliable labels.")

MIN_GAMES_FOR_LABEL = int(min_games)

status = st.empty()
try:
    status.info("Loading data…")
    tables = load_tables(data_dir)
    X, y, meta, label_source = build_dataset(tables, MIN_GAMES_FOR_LABEL)
    status.success(f"Loaded {len(X):,} players with labels across {y.nunique()} positions (labels from {label_source}).")
except Exception as e:
    st.error(f"Failed to load/build dataset: {e}")
    st.stop()

col1, col2 = st.columns([2.3, 1])
with col1:
    st.subheader("Train & Evaluate")
    model, acc, cm, report_txt, report_dict, test_idx, y_test, y_pred = train_model(X, y, model_choice)

    # Baseline accuracy (most frequent class)
    baseline = y.value_counts(normalize=True).max()

    st.markdown(f"**Overall Accuracy:** {acc:.3f}  \n"
                f"_Baseline (most frequent class):_ **{baseline:.3f}**")

    st.subheader("Confusion Matrix")
    st.caption(
        "Rows are **actual** positions, columns are **predicted** positions. "
        "High diagonal values mean the model gets those positions correct more often."
    )
    fig = plot_cm(cm, sorted(y.unique()))
    st.pyplot(fig)

    # Per-class metrics table
    st.subheader("Per-Class Performance")
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    # Top misclassifications
    st.subheader("Top Misclassifications")
    labels = sorted(y.unique())
    mis = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            if i != j and cm[i, j] > 0:
                mis.append({"actual": true_lab, "predicted": pred_lab, "count": int(cm[i, j])})
    mis_df = pd.DataFrame(mis).sort_values("count", ascending=False).head(10)
    if mis_df.empty:
        st.success("No misclassifications found on the test split 🎉")
    else:
        st.table(mis_df)

    if model_choice == "Decision Tree":
        st.subheader("Decision Tree (visual excerpt)")
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
buffer = BytesIO()
joblib.dump(model, buffer)
buffer.seek(0)
st.download_button("Download model.pkl", data=buffer, file_name="position_model.pkl")

with st.expander("What am I looking at?"):
    st.markdown(
        "- **Labels source**: The app auto-detects labels from **Appearances** (G_* columns) or falls back to **Fielding** (POS/G).\n"
        "- **Features**: Height, weight, and aggregated fielding stats.\n"
        "- **Accuracy vs Baseline**: If accuracy is only slightly above the baseline, the model might be overfitting or the task is imbalanced.\n"
        "- **Confusion Matrix**: Diagonal = correct predictions; off-diagonal = mistakes.\n"
        "- **Per-Class Metrics**: Precision/Recall/F1 for each position help diagnose imbalance.\n"
    )
""")

out_path = Path("/mnt/data/app.py")
out_path.write_text(app_code)
str(out_path)
