"""
CrediPredict - Machine Learning Training & Evaluation Script
Trains Logistic Regression, Decision Tree, and Random Forest models with 5-Fold Cross Validation
and GridSearchCV Hyperparameter Tuning. Saves the best model pipeline.
Written for B.Tech Data Science project.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

from data_loader import load_raw_data
from preprocessing import clean_data, engineer_features, build_preprocessor

# Define File Paths
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "loan_approval_model.pkl")
REPORT_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "model_comparison.csv")


def run_training_pipeline():
    print("=" * 60)
    print("CrediPredict - Starting Machine Learning Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    processed_df = engineer_features(clean_df)

    # 2. Features & Target
    X = processed_df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = processed_df['Loan_Status'].map({'Y': 1, 'N': 0})

    # 3. Train / Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 4. Build Preprocessor Pipeline
    preprocessor, num_cols, cat_cols = build_preprocessor()

    # 5. Define Candidate Models & Hyperparameter Grids
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'lbfgs']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'model__max_depth': [3, 5, 7, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7, None],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }
        }
    }

    results = []
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in model_configs.items():
        print(f"\n---> Training & Tuning {name}...")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])

        # Cross Validation Score (Baseline)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        mean_cv_acc = np.mean(cv_scores)
        print(f"     Mean 5-Fold CV Accuracy (Baseline): {mean_cv_acc:.4f}")

        # Hyperparameter Tuning with GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid=config['param_grid'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        trained_pipelines[name] = best_pipeline
        print(f"     Best Params: {grid_search.best_params_}")

        # Evaluate on Unseen Test Set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"     Test Performance -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        results.append({
            'Model': name,
            'CV_Mean_Accuracy': round(mean_cv_acc, 4),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })

    # 6. Save Comparison Report
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(REPORT_SAVE_PATH), exist_ok=True)
    results_df.to_csv(REPORT_SAVE_PATH, index=False)
    print("\n" + "=" * 60)
    print("Model Comparison Results:")
    print(results_df.to_string(index=False))
    print(f"\nComparison report saved to: {REPORT_SAVE_PATH}")

    # 7. Select Best Model based on F1 Score & Accuracy
    results_df_sorted = results_df.sort_values(by=['F1 Score', 'Accuracy'], ascending=False)
    best_model_name = results_df_sorted.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n--> Selected Best Model: '{best_model_name}'")

    # 8. Save Best Pipeline using joblib
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"Best model pipeline successfully saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)

    return best_model_name, results_df


if __name__ == "__main__":
    run_training_pipeline()
