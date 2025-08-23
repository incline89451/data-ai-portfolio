#!/usr/bin/env python
# coding: utf-8

# # 🏟️ Predict Future Hall of Famers (Lahman + scikit-learn)
# This notebook loads local Parquet files, builds features from batting/pitching stats, and trains models to predict Hall of Fame induction.

# In[5]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# ## 📂 Load Local Parquet Files

# In[6]:


people_path = "/Users/todd/Desktop/csvdatabase/People/People.parquet"
batting_path = "/Users/todd/Desktop/csvdatabase/Batting/Batting.parquet"
pitching_path = "/Users/todd/Desktop/csvdatabase/Pitching/Pitching.parquet"
hof_path = "/Users/todd/Desktop/csvdatabase/HallOfFame/HallOfFame.parquet"

people = pd.read_parquet(people_path)
batting = pd.read_parquet(batting_path)
pitching = pd.read_parquet(pitching_path)
hof = pd.read_parquet(hof_path)

people.head()


# ## ⚙️ Feature Engineering

# In[7]:


def agg_batting(df: pd.DataFrame) -> pd.DataFrame:
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

def agg_pitching(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["W","L","G","GS","CG","SHO","SV","IPouts","H","ER","HR","BB","SO","ERA","IBB","WP","HBP","BK","BFP"]
    agg = df.groupby("playerID", as_index=False)[num_cols].sum(min_count=1)
    agg["IP"] = agg["IPouts"] / 3.0
    agg["ERA_calc"] = 9 * agg["ER"] / agg["IP"].replace(0, np.nan)
    agg["WHIP"] = (agg["BB"] + agg["H"]) / agg["IP"].replace(0, np.nan)
    agg["K_per_9"] = 9 * agg["SO"] / agg["IP"].replace(0, np.nan)
    agg["BB_per_9"] = 9 * agg["BB"] / agg["IP"].replace(0, np.nan)
    return agg

def derive_people_features(df_people, batting_df, pitching_df):
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

def build_label(hof_df):
    keep = hof_df[hof_df["category"].str.lower() == "player"].copy()
    keep["inducted_flag"] = (keep["inducted"].astype(str).str.upper() == "Y").astype(int)
    inducted_any = keep.groupby("playerID")["inducted_flag"].max().reset_index().rename(columns={"inducted_flag":"HOF"})
    last_ballot_year = keep.groupby("playerID")["yearid"].max().reset_index().rename(columns={"yearID":"last_ballot_year"})
    return pd.merge(inducted_any, last_ballot_year, on="playerID", how="left")

bat_agg = agg_batting(batting)
pit_agg = agg_pitching(pitching)
ppl_feats = derive_people_features(people, batting, pitching)
labels = build_label(hof)

feats = ppl_feats.merge(bat_agg, on="playerID", how="left")
feats = feats.merge(pit_agg, on="playerID", how="left")
data = feats.merge(labels, on="playerID", how="inner")
data.head()


# ## 🤖 Train Models

# In[8]:


drop_cols = ["playerID","nameFirst","nameLast","debut","finalGame"]
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
y_pred_log = log_pipe.predict(X_test)
y_prob_log = log_pipe.predict_proba(X_test)[:,1]

print("Logistic Regression:")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

# Random Forest
rf_pipe = Pipeline([("clf", RandomForestClassifier(n_estimators=500, random_state=42))])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
y_prob_rf = rf_pipe.predict_proba(X_test)[:,1]

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))


# ## 📈 ROC Curves

# In[9]:


fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,6))
plt.plot(fpr_log, tpr_log, label="LogReg")
plt.plot(fpr_rf, tpr_rf, label="RandomForest")
plt.plot([0,1],[0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()


# ## 🔎 Feature Importance

# In[10]:


# Logistic coefficients
coefs = log_pipe.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": feature_cols, "coef": coefs}).sort_values("coef", ascending=False)
print("Top Logistic Regression Coefficients:")
display(coef_df.head(15))

# Random Forest importances
importances = rf_pipe.named_steps["clf"].feature_importances_
fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
print("Top Random Forest Features:")
display(fi_df.head(15))


# ## 🧮 Predict a Player’s HOF Probability

# In[11]:


# Build a mapping of player name to row
data["player_name"] = data["nameFirst"].fillna("") + " " + data["nameLast"].fillna("")
player_map = dict(zip(data["player_name"], data["playerID"]))

# Example: Try predicting Ken Griffey Jr. if exists
name_choice = "Ken Griffey Jr."

if name_choice in player_map:
    pid = player_map[name_choice]
    row = data[data["playerID"] == pid]
    feats_row = row[feature_cols].fillna(0)
    prob_rf = rf_pipe.predict_proba(feats_row)[:,1][0]
    prob_log = log_pipe.predict_proba(feats_row)[:,1][0]
    print(f"{name_choice} → Logistic={prob_log:.3f}, RandomForest={prob_rf:.3f}")
else:
    print(f"{name_choice} not in dataset.")

