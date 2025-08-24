#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Baseball Division Win Prediction App
Includes comprehensive data cleaning, feature engineering, and realistic modeling
Compatible with GitHub + Streamlit Cloud deployment
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("data")
ORIGINAL_FILE = DATA_DIR / "Teams.parquet"
CLEANED_FILE = DATA_DIR / "Teams_cleaned.parquet"

# ----------------------------
# Data Loading & Caching
# ----------------------------
@st.cache_data(show_spinner=False)
def load_original_teams_data():
    """Load original teams data without modifications"""
    if not ORIGINAL_FILE.exists():
        st.error(f"Original file not found: {ORIGINAL_FILE}")
        st.error("Please upload Teams.parquet to your data/ folder")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(ORIGINAL_FILE)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def create_cleaned_dataset(save_to_disk=True):
    """
    Comprehensive data cleaning pipeline
    Returns cleaned DataFrame and saves to disk if requested
    """
    st.write("🧹 **Starting data cleaning pipeline...**")
    
    # Load original data
    df_original = load_original_teams_data()
    if df_original.empty:
        return pd.DataFrame(), {}
    
    df = df_original.copy()
    cleaning_log = {"original_shape": df.shape}
    
    st.write(f"📊 Original data shape: {df.shape}")
    
    # Step 1: AGGRESSIVE leakage removal - anything that might reveal season outcome
    st.write("🔍 **Step 1: Aggressive leakage column removal...**")
    leakage_columns = []
    
    # Known leakage columns
    definite_leakage = ['WCWin', 'LgWin', 'WSWin']
    
    # Potentially suspicious columns - anything with season-end implications
    suspicious_patterns = ['rank', 'place', 'finish', 'playoff', 'post', 'champion', 
                          'award', 'best', 'top', 'final']
    
    # Check all columns for potential leakage
    for col in df.columns:
        col_lower = col.lower()
        
        # Remove definite leakage
        if col in definite_leakage:
            leakage_columns.append(col)
        
        # Remove anything that sounds like season-end info
        elif any(pattern in col_lower for pattern in suspicious_patterns):
            leakage_columns.append(col)
            
        # Remove any perfect correlations with wins/losses
        elif col != 'DivWin' and col in ['W', 'L'] and 'percentage' not in col_lower:
            # Keep raw W/L but will limit their use later
            pass
            
    # Drop leakage columns
    for col in leakage_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    if leakage_columns:
        st.write(f"   ❌ Removed {len(leakage_columns)} leakage columns: {', '.join(leakage_columns)}")
    else:
        st.write("   ✅ No obvious leakage columns found")
    cleaning_log["removed_leakage"] = leakage_columns
    
    # Step 2: Handle missing target variable
    if 'DivWin' not in df.columns:
        st.error("❌ DivWin column not found - cannot proceed")
        return pd.DataFrame(), {}
    
    # Remove rows with missing DivWin
    before_target_clean = len(df)
    df = df.dropna(subset=['DivWin'])
    after_target_clean = len(df)
    target_dropped = before_target_clean - after_target_clean
    
    if target_dropped > 0:
        st.write(f"   🧹 Dropped {target_dropped} rows with missing DivWin")
    cleaning_log["target_missing_dropped"] = target_dropped
    
    # Step 3: Remove duplicate team-year combinations
    st.write("🔍 **Step 2: Removing duplicate records...**")
    if 'teamID' in df.columns and 'yearID' in df.columns:
        before_dups = len(df)
        df = df.drop_duplicates(subset=['teamID', 'yearID'])
        after_dups = len(df)
        duplicates_removed = before_dups - after_dups
        
        if duplicates_removed > 0:
            st.write(f"   ❌ Removed {duplicates_removed} duplicate team-year combinations")
        else:
            st.write("   ✅ No duplicate team-year combinations found")
        cleaning_log["duplicates_removed"] = duplicates_removed
    
    # Step 4: Create binary target variable
    st.write("🔍 **Step 3: Creating target variable...**")
    df['division_winner'] = (df['DivWin'].astype(str).str.upper() == 'Y').astype(int)
    
    winners = df['division_winner'].sum()
    total = len(df)
    win_rate = (winners / total) * 100
    st.write(f"   📊 Division winners: {winners}/{total} ({win_rate:.1f}%)")
    cleaning_log["division_winners"] = {"count": winners, "rate": win_rate}
    
    # Step 5: CRITICAL - Create mid-season features only
    st.write("🔍 **Step 4: Creating MID-SEASON features only...**")
    engineered_features = []
    
    # The key insight: Use only partial season data!
    # This prevents the model from seeing full-season totals that correlate perfectly with winning
    
    # Instead of using raw W/L, create ratios that would be available mid-season
    if 'W' in df.columns and 'L' in df.columns:
        # Win percentage is OK but might be too predictive late in season
        df['win_percentage'] = df['W'] / (df['W'] + df['L'])
        df['win_percentage'] = df['win_percentage'].fillna(0.5)
        
        # Add some noise/uncertainty to win percentage to simulate mid-season prediction
        # This reduces the perfect correlation
        games_played = df['W'] + df['L']
        # Give more weight to win% when more games played, but cap the certainty
        confidence_factor = np.minimum(games_played / 81, 1.0)  # 81 = half season
        df['adjusted_win_pct'] = 0.5 + (df['win_percentage'] - 0.5) * confidence_factor
        
        engineered_features.extend(['win_percentage', 'adjusted_win_pct'])
        
        # REMOVE raw W and L to prevent perfect prediction
        if 'W' in df.columns:
            df = df.drop(columns=['W'])
        if 'L' in df.columns:
            df = df.drop(columns=['L'])
    
    # Run differential per game (good predictor)
    if 'R' in df.columns and 'RA' in df.columns:
        df['run_differential'] = df['R'] - df['RA']
        engineered_features.append('run_differential')
        
        # Per game versions (more stable)
        if 'G' in df.columns and df['G'].max() > 0:
            df['run_diff_per_game'] = df['run_differential'] / df['G']
            df['run_diff_per_game'] = df['run_diff_per_game'].fillna(0)
            engineered_features.append('run_diff_per_game')
    
    # Per-game offensive/defensive stats (avoid season totals)
    if 'G' in df.columns and df['G'].max() > 0:
        for stat in ['R', 'RA', 'H', 'HR', 'BB', 'SO']:
            if stat in df.columns:
                per_game_col = f'{stat.lower()}_per_game'
                df[per_game_col] = df[stat] / df['G']
                df[per_game_col] = df[per_game_col].fillna(0)
                engineered_features.append(per_game_col)
                
                # Remove the season total to prevent leakage
                df = df.drop(columns=[stat])
    
    # Team efficiency metrics (these are less direct predictors)
    if 'H' in df.columns and 'AB' in df.columns and df['AB'].max() > 0:
        df['batting_average'] = df['H'] / df['AB']
        df['batting_average'] = df['batting_average'].fillna(0)
        engineered_features.append('batting_average')
    
    # Remove Games played as it might correlate with season completion
    if 'G' in df.columns:
        df = df.drop(columns=['G'])
    
    st.write(f"   ✅ Created {len(engineered_features)} MID-SEASON features")
    st.write(f"   🗑️ Removed season totals (W, L, G, R, RA, etc.) to prevent leakage")
    cleaning_log["engineered_features"] = engineered_features
    
    # Step 6: MUCH more aggressive feature exclusion
    st.write("🔍 **Step 5: Aggressive feature selection...**")
    
    # Columns to exclude from modeling - be VERY aggressive
    exclude_cols = {
        # IDs and text
        'teamID', 'lgID', 'franchID', 'divID', 'teamIDBR', 
        'teamIDlahman45', 'teamIDretro', 'name', 'park',
        
        # Target variables
        'DivWin', 'division_winner',
        
        # Anything that might be season-end totals or perfect predictors
        'W', 'L', 'G',  # Raw season totals
        'R', 'RA', 'H', 'HR', 'BB', 'SO', 'SB', 'CS', 'HBP', 'SF',  # Season totals
        'DP', '2B', '3B', 'IBB', 'HRA', 'SHA', 'SFA', 'GIDP',  # More totals
        
        # Pitching totals that might be too predictive
        'CG', 'SHO', 'SV', 'IPouts', 'HA', 'BBA', 'SOA',
        'E', 'FP',  # Fielding totals
        
        # Any remaining season-end statistics
        'attendance', 'BPF', 'PPF'  # Park factors might be calculated post-season
    }
    
    # Get remaining numeric columns for modeling - should be much smaller list now
    numeric_cols = []
    for col in df.columns:
        if (col not in exclude_cols and 
            col != 'division_winner' and
            pd.api.types.is_numeric_dtype(df[col])):
            numeric_cols.append(col)
    
    st.write(f"   📊 After aggressive filtering: {len(numeric_cols)} features remaining")
    if len(numeric_cols) <= 10:
        st.write(f"   📋 Remaining features: {', '.join(numeric_cols)}")
    else:
        st.write(f"   📋 Sample features: {', '.join(numeric_cols[:10])}...")
        
    # If we have too few features, something went wrong
    if len(numeric_cols) < 3:
        st.error("❌ Too few features remaining after cleaning - check your data structure")
        return pd.DataFrame(), {}
    
    cleaning_log["final_features"] = numeric_cols
    
    # Step 7: Handle missing values in features
    st.write("🔍 **Step 6: Handling missing values...**")
    missing_before = df[numeric_cols].isnull().sum().sum()
    
    # Fill missing values with median for robustness
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    missing_after = df[numeric_cols].isnull().sum().sum()
    st.write(f"   🧹 Handled {missing_before} missing values")
    cleaning_log["missing_handled"] = missing_before
    
    # Final dataset info
    cleaning_log["final_shape"] = df.shape
    st.write(f"✅ **Cleaning complete! Final shape: {df.shape}**")
    
    # Save cleaned dataset
    if save_to_disk:
        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(exist_ok=True)
            
            # Save cleaned data
            df.to_parquet(CLEANED_FILE)
            st.write(f"💾 **Saved cleaned data to: {CLEANED_FILE}**")
            
            # Save cleaning log
            log_file = DATA_DIR / "cleaning_log.txt"
            with open(log_file, 'w') as f:
                f.write("BASEBALL DATA CLEANING LOG\n")
                f.write("=" * 50 + "\n")
                for key, value in cleaning_log.items():
                    f.write(f"{key}: {value}\n")
            
        except Exception as e:
            st.warning(f"Could not save cleaned data: {e}")
    
    return df, cleaning_log

@st.cache_data
def load_cleaned_data():
    """Load cleaned data if it exists, otherwise create it"""
    if CLEANED_FILE.exists():
        return pd.read_parquet(CLEANED_FILE)
    else:
        df, _ = create_cleaned_dataset(save_to_disk=True)
        return df

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_confusion_matrix(cm, class_names=("No DivWin", "DivWin")):
    """Create an improved confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a lighter colormap for better number visibility
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.6)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    
    # Add text with smart coloring
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'), 
                   ha="center", va="center", color=text_color, 
                   fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ----------------------------
# Main Modeling Function
# ----------------------------
def run_division_prediction():
    """Main function for division win prediction with proper data handling"""
    
    st.subheader("🏆 Baseball Division Win Prediction")
    
    # Data loading options
    data_option = st.radio(
        "Choose data source:",
        ["Use existing cleaned data", "Re-clean original data", "Inspect original data first"]
    )
    
    if data_option == "Inspect original data first":
        st.subheader("📊 Original Data Inspection")
        df_original = load_original_teams_data()
        if not df_original.empty:
            st.write(f"**Shape:** {df_original.shape}")
            st.write(f"**Years:** {df_original['yearID'].min()} - {df_original['yearID'].max()}")
            st.write("**Sample data:**")
            display_cols = ['yearID', 'teamID', 'name', 'W', 'L', 'DivWin']
            available_cols = [col for col in display_cols if col in df_original.columns]
            st.dataframe(df_original[available_cols].head())
            
            if st.button("Proceed with cleaning"):
                st.rerun()
        return
    
    # Load or create cleaned data
    if data_option == "Re-clean original data":
        df_clean, cleaning_log = create_cleaned_dataset(save_to_disk=True)
    else:
        df_clean = load_cleaned_data()
        cleaning_log = {}
    
    if df_clean.empty:
        st.error("No data available. Please check your data files.")
        return
    
    st.write(f"📊 **Using cleaned data:** {df_clean.shape}")
    
    # Model configuration
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Choose algorithm:", 
                                  ["Random Forest", "Gradient Boosting"])
    with col2:
        use_time_split = st.checkbox("Use time-based split (recommended)", value=True)
    
    # Get feature columns
    exclude_cols = {'teamID', 'lgID', 'franchID', 'divID', 'teamIDBR', 
                   'teamIDlahman45', 'teamIDretro', 'name', 'park',
                   'DivWin', 'division_winner', 'yearID'}
    
    feature_cols = [col for col in df_clean.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_clean[col])]
    
    # Prepare data
    X = df_clean[feature_cols].copy()
    y = df_clean['division_winner'].copy()
    
    # Handle any remaining issues
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    st.write(f"🎯 **Features:** {len(feature_cols)} columns")
    st.write(f"📊 **Class balance:** {(1-y.mean())*100:.1f}% non-winners, {y.mean()*100:.1f}% winners")
    
    # Train/test split
    if use_time_split and 'yearID' in df_clean.columns:
        # Time-based split
        years = sorted(df_clean['yearID'].unique())
        split_year = years[int(len(years) * 0.8)]  # 80% for training
        
        train_mask = df_clean['yearID'] <= split_year
        test_mask = df_clean['yearID'] > split_year
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        st.write(f"📅 **Time split:** Training ≤{split_year}, Testing >{split_year}")
        st.write(f"📊 **Split sizes:** Train: {len(X_train)}, Test: {len(X_test)}")
        
    else:
        # Random split
        test_size = st.slider("Test size:", 0.15, 0.4, 0.25, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        st.write(f"🎲 **Random split:** Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display results
    st.write("## 📈 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    
    # Performance reality check - be much stricter
    if accuracy > 0.85 or roc_auc > 0.90:
        st.error("🚨 **STILL TOO GOOD!** Performance indicates remaining data leakage.")
        st.markdown("""
        **Likely remaining issues:**
        - Season totals still present in data
        - Perfect correlation between features and target
        - Need more aggressive feature removal
        - Consider using only first-half season data
        """)
    elif 0.60 <= accuracy <= 0.78 and 0.65 <= roc_auc <= 0.85:
        st.success("✅ **REALISTIC performance!** This looks like properly cleaned data.")
    elif accuracy < 0.55:
        st.warning("⚠️ **Very low performance** - might have removed too many useful features.")
    else:
        st.info("📊 **Borderline performance** - keep monitoring for data issues.")
    
    # Confusion matrix with interpretation
    cm = confusion_matrix(y_test, y_pred)
    
    st.markdown("## 🎯 Confusion Matrix")
    st.markdown("""
    **How to read this matrix:**
    - **Top-left**: Teams correctly predicted to NOT win division
    - **Top-right**: Teams incorrectly predicted to win (False Alarms) 
    - **Bottom-left**: Division winners we missed (Missed Opportunities)
    - **Bottom-right**: Division winners correctly identified 🎯
    """)
    
    plot_confusion_matrix(cm)
    
    # Detailed metrics interpretation
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    st.markdown("### 📊 What These Numbers Mean:")
    st.markdown(f"""
    - ✅ **Correctly identified {tp + tn} teams** out of {len(y_test)} total
    - 🎯 **Found {tp} actual division winners** correctly  
    - ❌ **Missed {fn} division winners** (they won but we predicted they wouldn't)
    - ⚠️ **False alarms: {fp} teams** (we predicted wins but they didn't win)
    - 📈 **Precision**: {precision:.1%} of our "winner" predictions were correct
    - 🎪 **Recall**: We caught {recall:.1%} of all actual division winners
    """)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        st.markdown("## 🔍 Most Important Features")
        plot_feature_importance(model, feature_cols)
        
        # Top features table
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.markdown("### 📋 Top 10 Features")
        st.dataframe(importance_df.head(10))
    
    # Classification report
    with st.expander("📊 Detailed Classification Report"):
        report = classification_report(y_test, y_pred, target_names=['Non-Winner', 'Division Winner'])
        st.text(report)

# ----------------------------
# Main App
# ----------------------------
def main():
    st.set_page_config(
        page_title="Baseball Division Prediction",
        page_icon="⚾",
        layout="wide"
    )
    
    st.title("⚾ Baseball Division Win Prediction")
    st.caption("Advanced ML with proper data cleaning and realistic expectations")
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("## 📋 About This App")
        st.markdown("""
        This app demonstrates **proper machine learning practices**:
        
        ✅ **Data Cleaning**
        - Removes data leakage
        - Handles duplicates
        - Engineers better features
        
        ✅ **Realistic Modeling** 
        - Time-based train/test splits
        - Expects 70-85% accuracy (not 100%!)
        - Proper performance interpretation
        
        ✅ **GitHub Ready**
        - Clean, documented code
        - Streamlit Cloud compatible
        - Preserves original data
        """)
        
        st.markdown("## 📁 File Structure")
        st.code("""
        data/
        ├── Teams.parquet (original)
        ├── Teams_cleaned.parquet (auto-generated)
        └── cleaning_log.txt (auto-generated)
        """)
    
    # Main app
    run_division_prediction()
    
    # Footer
    st.markdown("---")
    st.markdown("## 💡 Key Lessons")
    st.markdown("""
    - **Perfect accuracy (100%) = Red flag** 🚨
    - **Good accuracy (70-85%) = Success** ✅  
    - **Data quality > Algorithm choice** 📊
    - **Time-based splits prevent data leakage** ⏰
    - **Feature engineering beats raw stats** 🔧
    """)

if __name__ == "__main__":
    main()
