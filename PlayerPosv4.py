
# app.py - Player Position Optimization
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

DATA_DIR = "data"
POS_COLUMNS = [("g_p","P"),("g_c","C"),("g_1b","1B"),("g_2b","2B"),("g_3b","3B"),("g_ss","SS"),("g_lf","LF"),("g_cf","CF"),("g_rf","RF")]
MIN_GAMES_DEFAULT = 20
RANDOM_STATE = 42

PARQUET_NAMES = {
    "people": ["People.parquet","people.parquet"],
    "appearances": ["Appearances.parquet","appearances.parquet"],
    "fielding": ["Fielding.parquet","fielding.parquet"],
}

VALID_POS = ["P","C","1B","2B","3B","SS","LF","CF","RF"]

def find_file(data_dir:str,candidates:List[str])->str:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    existing = {f.lower():os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.lower().endswith(".parquet")}
    for name in candidates:
        p = existing.get(name.lower())
        if p: return p
    for name in candidates:
        key = name.lower().replace(".parquet","")
        for k,v in existing.items():
            if key in k: return v
    raise FileNotFoundError(f"None of {candidates} found in {data_dir}.")

@st.cache_data(show_spinner=False)
def load_tables(data_dir:str)->Dict[str,pd.DataFrame]:
    people = pd.read_parquet(find_file(data_dir,PARQUET_NAMES["people"]),engine="pyarrow")
    app = pd.read_parquet(find_file(data_dir,PARQUET_NAMES["appearances"]),engine="pyarrow")
    fld = pd.read_parquet(find_file(data_dir,PARQUET_NAMES["fielding"]),engine="pyarrow")
    return {"people":people,"appearances":app,"fielding":fld}

def derive_primary_from_appearances(app:pd.DataFrame,min_games:int)->pd.DataFrame:
    cols_lower = {c.lower():c for c in app.columns}
    present = [(cols_lower[k],short) for k,short in POS_COLUMNS if k in cols_lower]
    if not present: return pd.DataFrame(columns=["playerID","primary_pos","total_games_tracked"])
    keep_cols = ["playerID"]+[orig for orig,_ in present]
    app2 = app[keep_cols].copy()
    for orig,_ in present: app2[orig]=pd.to_numeric(app2[orig],errors="coerce").fillna(0)
    agg = app2.groupby("playerID",as_index=False).sum(numeric_only=True)
    pos_counts = [agg[orig] for orig,_ in present]
    short_labels = [short for _,short in present]
    pos_mat = np.vstack(pos_counts).T
    total_games = pos_mat.sum(axis=1)
    primary_idx = pos_mat.argmax(axis=1)
    labels = pd.Series([short_labels[i] for i in primary_idx],index=agg.index,name="primary_pos")
    out = agg[["playerID"]].copy()
    out["total_games_tracked"]=total_games
    out["primary_pos"]=labels
    out = out[out["total_games_tracked"]>=min_games].copy()
    return out

def derive_primary_from_fielding(fld:pd.DataFrame,min_games:int)->pd.DataFrame:
    cols_lower={c.lower():c for c in fld.columns}
    if "pos" not in cols_lower or "g" not in cols_lower: return pd.DataFrame(columns=["playerID","primary_pos","total_games_tracked"])
    fld2 = fld[["playerID",cols_lower["pos"],cols_lower["g"]]].copy()
    fld2[cols_lower["g"]] = pd.to_numeric(fld2[cols_lower["g"]],errors="coerce").fillna(0)
    fld2 = fld2[fld2[cols_lower["pos"]].isin(VALID_POS)]
    gp = fld2.groupby(["playerID",cols_lower["pos"]],as_index=False)[cols_lower["g"]].sum()
    totals = gp.groupby("playerID",as_index=False)[cols_lower["g"]].sum().rename(columns={cols_lower["g"]:"total_games_tracked"})
    idx = gp.groupby("playerID")[cols_lower["g"]].idxmax()
    primary = gp.loc[idx,["playerID",cols_lower["pos"]]].rename(columns={cols_lower["pos"]:"primary_pos"})
    out = primary.merge(totals,on="playerID",how="left")
    out = out[out["total_games_tracked"]>=min_games].copy()
    return out

def build_dataset(tables:Dict[str,pd.DataFrame],min_games:int)->Tuple[pd.DataFrame,pd.Series,pd.DataFrame,str]:
    people = tables["people"][["playerID","height","weight","nameFirst","nameLast"]].copy()
    for col in ["height","weight"]: people[col]=pd.to_numeric(people[col],errors="coerce")
    labels = derive_primary_from_appearances(tables["appearances"],min_games)
    source="Appearances"
    if labels.empty:
        labels = derive_primary_from_fielding(tables["fielding"],min_games)
        source="Fielding"
    if labels.empty: raise ValueError("No primary positions could be derived")
    fld = tables["fielding"].copy()
    num_cols = [c for c in ["G","GS","InnOuts","PO","A","E","DP"] if c in fld.columns]
    fld_numeric = fld[["playerID"]+num_cols].copy()
    for c in num_cols: fld_numeric[c]=pd.to_numeric(fld_numeric[c],errors="coerce")
    fld_agg = fld_numeric.groupby("playerID",as_index=False).sum(numeric_only=True).rename(columns={
        "G":"fld_G","GS":"fld_GS","InnOuts":"fld_InnOuts","PO":"fld_PO","A":"fld_A","E":"fld_E","DP":"fld_DP"
    })
    if set(["fld_PO","fld_A","fld_E"]).issubset(fld_agg.columns):
        fld_agg["fld_chances"]=fld_agg[["fld_PO","fld_A","fld_E"]].sum(axis=1)
        fld_agg["fld_fpct"]=(fld_agg["fld_PO"]+fld_agg["fld_A"])/fld_agg["fld_chances"].replace(0,np.nan)
    else:
        fld_agg["fld_chances"]=np.nan
        fld_agg["fld_fpct"]=np.nan
    if "fld_DP" in fld_agg.columns and "fld_G" in fld_agg.columns:
        fld_agg["fld_dp_per_g"]=fld_agg["fld_DP"]/fld_agg["fld_G"].replace(0,np.nan)
    else: fld_agg["fld_dp_per_g"]=np.nan
    feats = labels.merge(people,on="playerID",how="left").merge(fld_agg,on="playerID",how="left")
    candidate_features=["height","weight","fld_G","fld_GS","fld_InnOuts","fld_PO","fld_A","fld_E","fld_DP","fld_chances","fld_fpct","fld_dp_per_g"]
    feature_cols=[c for c in candidate_features if c in feats.columns]
    X = feats[feature_cols].copy().replace([np.inf,-np.inf],np.nan).fillna(0)
    y = feats["primary_pos"].copy()
    meta = feats[["playerID","primary_pos","total_games_tracked","nameFirst","nameLast"]].copy()
    return X,y,meta,source

def train_model(X:pd.DataFrame,y:pd.Series,model_name:str):
    if model_name=="SVM (rbf)":
        pipe = Pipeline([("scaler",StandardScaler()),("clf",SVC(kernel="rbf",probability=True,random_state=RANDOM_STATE))])
    elif model_name=="Decision Tree":
        pipe = Pipeline([("clf",DecisionTreeClassifier(max_depth=8,random_state=RANDOM_STATE))])
    else: raise ValueError("Unknown model: "+model_name)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)
    model=pipe.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred,labels=sorted(y.unique()))
    report = classification_report(y_test,y_pred,labels=sorted(y.unique()),output_dict=False)
    return model,acc,cm,report,X_test.index,y_test,y_pred

def plot_cm(cm:np.ndarray,class_labels:List[str]):
    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_labels,yticklabels=class_labels,ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45,ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig

st.set_page_config(page_title="Player Position Optimization",layout="wide")
st.title("⚾ Player Position Optimization")
st.caption("Classify players into their most effective positions using fielding stats and physical attributes (Lahman schemas).")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Data directory",value=DATA_DIR)
    model_choice = st.selectbox("Model",["SVM (rbf)","Decision Tree"])
    min_games = st.slider("Minimum total games to define a primary position",0,200,MIN_GAMES_DEFAULT,5)

MIN_GAMES_DEFAULT = int(min_games)

status = st.empty()
try:
    status.info("Loading data…")
    tables = load_tables(data_dir)
    X,y,meta,source=build_dataset(tables,MIN_GAMES_DEFAULT)
    status.success(f"Loaded {len(X):,} players using {source} table across {y.nunique()} positions")
except Exception as e:
    st.error(f"Failed to load/build dataset: {e}")
    st.stop()

col1,col2 = st.columns([2,1])
with col1:
    st.subheader("Train & Evaluate")
    model,acc,cm,report,test_idx,y_test,y_pred = train_model(X,y,model_choice)
    st.write(f"**Accuracy:** {acc:.3f}")
    fig = plot_cm(cm,sorted(y.unique()))
    st.pyplot(fig)
    if model_choice=="Decision Tree":
        st.subheader("Decision Tree (visual excerpt)")
        try:
            tree_clf = model.named_steps.get("clf",None) or model
            fig2,ax2=plt.subplots(figsize=(12,8))
            plot_tree(tree_clf,max_depth=3,fontsize=8,feature_names=list(X.columns),class_names=sorted(y.unique()),filled=False)
            st.pyplot(fig2)
        except Exception as ex:
            st.info(f"Tree visualization unavailable: {ex}")

with col2:
    st.subheader("Predict for a Player")
    test_meta = meta.loc[test_idx].copy()
    merged = test_meta.copy()
    merged["true"]=y_test.values
    merged["pred"]=y_pred
    merged["correct"]=(merged["true"]==merged["pred"])
    merged["display_name"]=merged["nameLast"]+", "+merged["nameFirst"]+" ("+merged["playerID"]+")"
    if len(merged)>0:
        player_row = st.selectbox("Choose a test-set player",merged["display_name"].tolist())
        row = merged[merged["display_name"]==player_row].iloc[0]
        st.metric("True position",row["true"])
        st.metric("Predicted position",row["pred"])
        st.write(f"Total games considered: {int(row['total_games_tracked'])}")
        st.write(f"Correct? {'✅' if row['correct'] else '❌'}")
    else:
        st.info("No players available in test set.")

st.divider()
st.subheader("Download trained model (optional)")
model_bytes = BytesIO()
joblib.dump(model,model_bytes)
model_bytes.seek(0)
st.download_button("Download model.pkl",data=model_bytes,file_name="position_model.pkl")
st.caption("Data schemas follow Lahman Baseball Database (People, Appearances, Fielding tables).")
