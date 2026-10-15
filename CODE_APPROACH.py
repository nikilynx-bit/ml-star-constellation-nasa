"""
CODE_APPROACH.py
ML Star Constellation NASA - Code-Based Approach

This script demonstrates the complete AI Project Lifecycle for classifying 
astronomical objects (Stars, Galaxies, Quasars) using tabular telescope data.

Lifecycle Steps:
1. Scoping the Project
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. AI Modelling
5. Evaluation

Dataset Required: 'Star_classification.csv' from Kaggle (deepaksingh1012/star-classification)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SCOPING THE PROJECT
# ==============================================================================
def scope_project():
    """Define the problem, target, and metrics."""
    scope = {
        "Project": "ML Star Constellation NASA",
        "Objective": "Classify astronomical objects based on sky-survey features.",
        "Target Variable": "class (STAR, GALAXY, QUASAR)",
        "Primary Metric": "Macro F1-Score (handles class imbalance)",
        "Models": ["DummyClassifier (Baseline)", "RandomForest", "HistGradientBoosting"]
    }
    print("✅ Step 1: Scoping Complete")
    for k, v in scope.items():
        print(f"   - {k}: {v}")
    return scope

# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
def clean_data(file_path="Star_classification.csv"):
    """Load, profile, and clean the astronomical dataset."""
    print("\n✅ Step 2: Data Cleaning...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please download the dataset from Kaggle.")
        return None

    # Drop astronomical ID columns that act as noise for ML
    id_cols = ['obj_ID', 'spec_obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'plate', 'MJD', 'fiber_ID']
    df = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
    
    # Drop rows with missing target values
    df = df.dropna(subset=['class'])
    
    # Convert target to string and strip whitespace
    df['class'] = df['class'].astype(str).str.strip()
    
    # Drop constant columns
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    
    print(f"   - Final Dataset Shape: {df.shape}")
    return df

# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def analyze_data(df):
    """Perform basic EDA and generate visualizations."""
    print("\n✅ Step 3: Exploratory Data Analysis (EDA)...")
    
    # Target Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(y=df['class'], order=df['class'].value_counts().index, palette="viridis")
    plt.title("Distribution of Astronomical Classes")
    plt.show()
    
    # Correlation Heatmap (Numeric features only)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.show()

# ==============================================================================
# 4. AI MODELLING
# ==============================================================================
def build_models(df):
    """Create preprocessing pipelines and train ML models."""
    print("\n✅ Step 4: AI Modelling...")
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing Pipeline
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define Models
    models = {
        "Baseline (Dummy)": DummyClassifier(strategy="stratified", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Create pipeline
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Train and Evaluate
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "y_pred": y_pred
        }
        print(f"   - Trained: {name}")
        
    return results, y_test

# ==============================================================================
# 5. EVALUATION
# ==============================================================================
def evaluate_models(results, y_test):
    """Compare model performance and visualize confusion matrices."""
    print("\n✅ Step 5: Evaluation...")
    
    print("\n--- Model Comparison ---")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['accuracy']:.4f}     | {metrics['f1_macro']:.4f}")
        
    # Plot Confusion Matrix for the best model (usually Hist Gradient Boosting)
    best_model_name = max(results, key=lambda k: results[k]['f1_macro'])
    best_preds = results[best_model_name]['y_pred']
    
    cm = confusion_matrix(y_test, best_preds, labels=np.unique(y_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("ML STAR CONSTELLATION NASA - AI PROJECT LIFECYCLE")
    print("="*60)
    
    scope = scope_project()
    df = clean_data()
    
    if df is not None:
        analyze_data(df)
        results, y_test = build_models(df)
        evaluate_models(results, y_test)
        print("\n✅ AI Project Cycle Complete!")
    else:
        print("\n❌ Pipeline halted due to missing dataset.")
