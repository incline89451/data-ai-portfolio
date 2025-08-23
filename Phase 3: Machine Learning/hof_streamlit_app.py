import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Configure page
st.set_page_config(
    page_title="⚾ Hall of Fame Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ Hall of Fame Predictor")
st.markdown("### Predict Future Hall of Famers using Machine Learning")

# Cache data loading and processing functions
@st.cache_data
def load_data():
    """Load local Parquet files"""
    try:
        people_path = "/Users/todd/Desktop/csvdatabase/People/People.parquet"
        batting_path = "/Users/todd/Desktop/csvdatabase/Batting/Batting.parquet"
        pitching_path = "/Users/todd/Desktop/csvdatabase/Pitching/Pitching.parquet"
        hof_path = "/Users/todd/Desktop/csvdatabase/HallOfFame/HallOfFame.parquet"
        
        people = pd.read_parquet(people_path)
        batting = pd.read_parquet(batting_path)
        pitching = pd.read_parquet(pitching_path)
        hof = pd.read_parquet(hof_path)
        
        return people, batting, pitching, hof
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}")
        st.stop()

@st.cache_data
def agg_batting(df):
    """Aggregate batting statistics"""
    num_cols = ["G","AB","R","H","2B","3B","HR","RBI","SB","CS","BB","SO","IBB","HBP","SH","SF","GIDP"]
    agg = df.groupby("playerID", as_index=False)[num_cols].sum(min_count=1)
    agg["BA"] = agg["H"] / agg["AB"].replace(0, np.nan)
    agg["OBP"] = (agg["H"] + agg["BB"].fillna(0) + agg["HBP"].fillna(0)) / (
        agg["AB"] + agg["BB"].fillna(0) + agg["HBP"].fillna(0) + agg["SF"].fillna(0)
    )
    singles = agg["H"] - agg["2B"].fillna(0) - agg["3B"].fillna(0) - agg["HR"].fillna(0)
    agg["SLG"] = (singles + 2*agg["2B"].fillna(0) + 3*agg["3B"].fillna(0) + 4*agg["HR"].fillna(0)) / agg["AB"].replace(0, np.nan)
    agg["OPS"] = agg["OBP"] + agg["SLG"]
    return agg

@st.cache_data
def agg_pitching(df):
    """Aggregate pitching statistics"""
    num_cols = ["W","L","G","GS","CG","SHO","SV","IPouts","H","ER","HR","BB","SO","ERA","IBB","WP","HBP","BK","BFP"]
    agg = df.groupby("playerID", as_index=False)[num_cols].sum(min_count=1)
    agg["IP"] = agg["IPouts"] / 3.0
    agg["ERA_calc"] = 9 * agg["ER"] / agg["IP"].replace(0, np.nan)
    agg["WHIP"] = (agg["BB"] + agg["H"]) / agg["IP"].replace(0, np.nan)
    agg["K_per_9"] = 9 * agg["SO"] / agg["IP"].replace(0, np.nan)
    agg["BB_per_9"] = 9 * agg["BB"] / agg["IP"].replace(0, np.nan)
    return agg

@st.cache_data
def derive_people_features(df_people, batting_df, pitching_df):
    """Derive features from people data"""
    bat_years = batting_df.groupby("playerID")["yearID"].agg(first_bat_year="min", last_bat_year="max").reset_index()
    pit_years = pitching_df.groupby("playerID")["yearID"].agg(first_pit_year="min", last_pit_year="max").reset_index()
    yrs = pd.merge(bat_years, pit_years, on="playerID", how="outer")
    yrs["first_year"] = yrs[["first_bat_year", "first_pit_year"]].min(axis=1)
    yrs["last_year"] = yrs[["last_bat_year", "last_pit_year"]].max(axis=1)
    yrs = yrs[["playerID","first_year","last_year"]]
    ppl = df_people[["playerID","nameFirst","nameLast","birthYear","debut","finalGame"]].copy()
    ppl = pd.merge(ppl, yrs, on="playerID", how="left")
    ppl["career_length"] = (ppl["last_year"] - ppl["first_year"]).astype("float")
    return ppl

@st.cache_data
def build_label(hof_df):
    """Build Hall of Fame labels"""
    keep = hof_df[hof_df["category"].str.lower() == "player"].copy()
    keep["inducted_flag"] = (keep["inducted"].astype(str).str.upper() == "Y").astype(int)
    inducted_any = keep.groupby("playerID")["inducted_flag"].max().reset_index().rename(columns={"inducted_flag":"HOF"})
    last_ballot_year = keep.groupby("playerID")["yearID"].max().reset_index().rename(columns={"yearID":"last_ballot_year"})
    return pd.merge(inducted_any, last_ballot_year, on="playerID", how="left")

@st.cache_data
def prepare_data():
    """Load and prepare all data"""
    people, batting, pitching, hof = load_data()
    
    bat_agg = agg_batting(batting)
    pit_agg = agg_pitching(pitching)
    ppl_feats = derive_people_features(people, batting, pitching)
    labels = build_label(hof)
    
    feats = ppl_feats.merge(bat_agg, on="playerID", how="left")
    feats = feats.merge(pit_agg, on="playerID", how="left")
    data = feats.merge(labels, on="playerID", how="inner")
    
    return data

@st.cache_resource
def train_models(data):
    """Train machine learning models"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    target_col = "HOF"
    feature_cols = [c for c in numeric_cols if c not in [target_col]]
    
    X = data[feature_cols].fillna(0)
    y = data[target_col].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Logistic Regression
    log_pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    log_pipe.fit(X_train, y_train)
    
    # Random Forest
    rf_pipe = Pipeline([("clf", RandomForestClassifier(n_estimators=500, random_state=42))])
    rf_pipe.fit(X_train, y_train)
    
    return log_pipe, rf_pipe, X_test, y_test, feature_cols

# Load data and train models
with st.spinner("Loading data and training models..."):
    data = prepare_data()
    log_pipe, rf_pipe, X_test, y_test, feature_cols = train_models(data)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["🏠 Overview", "📊 Model Performance", "🔍 Feature Importance", "⚾ Player Predictions"])

if page == "🏠 Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(data))
    with col2:
        hof_players = data["HOF"].sum()
        st.metric("Hall of Famers", hof_players)
    with col3:
        hof_rate = (hof_players / len(data)) * 100
        st.metric("HOF Rate", f"{hof_rate:.1f}%")
    with col4:
        st.metric("Features", len(feature_cols))
    
    st.subheader("Sample Data")
    display_cols = ["nameFirst", "nameLast", "career_length", "H", "HR", "RBI", "W", "SO", "HOF"]
    available_cols = [col for col in display_cols if col in data.columns]
    st.dataframe(data[available_cols].head(10))

elif page == "📊 Model Performance":
    st.header("Model Performance")
    
    # Get predictions
    y_pred_log = log_pipe.predict(X_test)
    y_prob_log = log_pipe.predict_proba(X_test)[:,1]
    y_pred_rf = rf_pipe.predict(X_test)
    y_prob_rf = rf_pipe.predict_proba(X_test)[:,1]
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        log_auc = roc_auc_score(y_test, y_prob_log)
        log_acc = accuracy_score(y_test, y_pred_log)
        log_precision = precision_score(y_test, y_pred_log)
        log_recall = recall_score(y_test, y_pred_log)
        
        st.metric("ROC-AUC", f"{log_auc:.3f}")
        st.metric("Accuracy", f"{log_acc:.3f}")
        st.metric("Precision", f"{log_precision:.3f}")
        st.metric("Recall", f"{log_recall:.3f}")
    
    with col2:
        st.subheader("Random Forest")
        rf_auc = roc_auc_score(y_test, y_prob_rf)
        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_precision = precision_score(y_test, y_pred_rf)
        rf_recall = recall_score(y_test, y_pred_rf)
        
        st.metric("ROC-AUC", f"{rf_auc:.3f}")
        st.metric("Accuracy", f"{rf_acc:.3f}")
        st.metric("Precision", f"{rf_precision:.3f}")
        st.metric("Recall", f"{rf_recall:.3f}")
    
    # ROC Curves
    st.subheader("ROC Curves")
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_log, y=tpr_log, name=f'Logistic Regression (AUC={log_auc:.3f})', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f'Random Forest (AUC={rf_auc:.3f})', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    st.plotly_chart(fig)

elif page == "🔍 Feature Importance":
    st.header("Feature Importance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression Coefficients")
        coefs = log_pipe.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": coefs}).sort_values("Coefficient", ascending=False)
        
        fig = px.bar(coef_df.head(15), x="Coefficient", y="Feature", orientation='h',
                    title="Top 15 Features (Logistic Regression)")
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
        st.dataframe(coef_df.head(10))
    
    with col2:
        st.subheader("Random Forest Feature Importance")
        importances = rf_pipe.named_steps["clf"].feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False)
        
        fig = px.bar(fi_df.head(15), x="Importance", y="Feature", orientation='h',
                    title="Top 15 Features (Random Forest)")
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
        st.dataframe(fi_df.head(10))

elif page == "⚾ Player Predictions":
    st.header("Player Hall of Fame Predictions")
    
    # Build player mapping
    data["player_name"] = data["nameFirst"].fillna("") + " " + data["nameLast"].fillna("")
    player_names = sorted(data["player_name"].tolist())
    
    # Player selection
    selected_player = st.selectbox("Select a player:", player_names, index=0)
    
    if selected_player:
        # Get player data
        player_data = data[data["player_name"] == selected_player].iloc[0]
        player_features = data[data["player_name"] == selected_player][feature_cols].fillna(0)
        
        # Make predictions
        prob_log = log_pipe.predict_proba(player_features)[:,1][0]
        prob_rf = rf_pipe.predict_proba(player_features)[:,1][0]
        actual_hof = player_data["HOF"]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Logistic Regression Probability", f"{prob_log:.3f}")
        with col2:
            st.metric("Random Forest Probability", f"{prob_rf:.3f}")
        with col3:
            hof_status = "Yes" if actual_hof == 1 else "No"
            st.metric("Actual HOF Status", hof_status)
        
        # Player stats
        st.subheader(f"{selected_player} - Career Statistics")
        
        stats_to_show = ["career_length", "G", "AB", "H", "HR", "RBI", "BA", "OPS", "W", "L", "IP", "SO", "ERA_calc"]
        available_stats = {stat: player_data.get(stat, "N/A") for stat in stats_to_show if stat in player_data.index}
        
        cols = st.columns(4)
        for i, (stat, value) in enumerate(available_stats.items()):
            with cols[i % 4]:
                if pd.notna(value) and value != "N/A":
                    if isinstance(value, float):
                        st.metric(stat.replace("_", " ").title(), f"{value:.3f}")
                    else:
                        st.metric(stat.replace("_", " ").title(), value)
                else:
                    st.metric(stat.replace("_", " ").title(), "N/A")
    
    # Batch predictions
    st.subheader("Top Predicted Hall of Famers (Not Yet Inducted)")
    
    non_hof = data[data["HOF"] == 0].copy()
    if len(non_hof) > 0:
        non_hof_features = non_hof[feature_cols].fillna(0)
        non_hof["rf_prob"] = rf_pipe.predict_proba(non_hof_features)[:,1]
        non_hof["log_prob"] = log_pipe.predict_proba(non_hof_features)[:,1]
        
        top_candidates = non_hof.nlargest(10, "rf_prob")[["player_name", "rf_prob", "log_prob"]]
        top_candidates.columns = ["Player", "Random Forest Prob", "Logistic Regression Prob"]
        st.dataframe(top_candidates)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source:** Lahman Baseball Database")
st.sidebar.markdown("**Models:** Logistic Regression & Random Forest")