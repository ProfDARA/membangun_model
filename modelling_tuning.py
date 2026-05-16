"""
Model Training with Hyperparameter Tuning - Kriteria 2: Skilled Level (3 Poin)
Uses MLflow Tracking UI (local) with MANUAL logging (not autolog)
Includes hyperparameter tuning and additional metrics
Author: Danang Agung Restu Aji
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(daily_csv: str = 'Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_sales_forecasting.csv'):
    """
    Load the cleaned daily sales forecasting dataset and prepare
    train/validation/test splits for regression.

    Args:
        daily_csv: Path to `daily_sales_forecasting.csv` produced by preprocessing

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (all pandas DataFrames/Series)
    """
    df = pd.read_csv(daily_csv, parse_dates=['Date'])

    # Ensure expected features exist
    expected_features = [
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7',
        'day', 'month', 'year', 'weekday', 'weekofyear'
    ]

    available = [c for c in expected_features if c in df.columns]
    if len(available) < 3:
        raise ValueError(f"Not enough feature columns found in {daily_csv}. Available: {available}")

    df = df.dropna().sort_values('Date')

    feature_cols = available
    target_col = 'Daily_Revenue'

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split: 70% train, 15% val, 15% test
    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train = X.iloc[:n_train].reset_index(drop=True)
    y_train = y.iloc[:n_train].reset_index(drop=True)

    X_val = X.iloc[n_train:n_train + n_val].reset_index(drop=True)
    y_val = y.iloc[n_train:n_train + n_val].reset_index(drop=True)

    X_test = X.iloc[n_train + n_val:].reset_index(drop=True)
    y_test = y.iloc[n_train + n_val:].reset_index(drop=True)

    print(f"Loaded daily sales forecasting data: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_regression(y_true, y_pred, dataset_name: str = ""):
    """
    Evaluate regression predictions with common metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # MAPE - handle zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = (np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))).mean() * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape_pct': mape
    }

    print(f"\n{dataset_name} Regression Metrics:")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return metrics


def hyperparameter_tuning_random_forest(X_train, y_train, X_val, y_val, X_test, y_test,
                                        output_dir: str = 'Membangun_model/artifacts'):
    """
    Hyperparameter tuning for RandomForestRegressor using RandomizedSearchCV
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING - RANDOM FOREST (REGRESSION)")
    print("=" * 80 + "\n")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'auto']
    }

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = RandomizedSearchCV(
        rf_base, param_grid, n_iter=20, cv=3, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=1, random_state=42
    )

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    print("Running hyperparameter tuning (20 iterations)...")
    grid_search.fit(X_combined, y_combined)

    best_model = grid_search.best_estimator_

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate
    train_metrics = evaluate_regression(y_train, y_train_pred, "TRAINING - Tuned RF")
    test_metrics = evaluate_regression(y_test, y_test_pred, "TEST - Tuned RF")

    return best_model, grid_search, train_metrics, test_metrics


def hyperparameter_tuning_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test,
                                           output_dir: str = 'Membangun_model/artifacts'):
    """
    Hyperparameter tuning for GradientBoostingRegressor
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING - GRADIENT BOOSTING (REGRESSION)")
    print("=" * 80 + "\n")

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10]
    }

    gb_base = GradientBoostingRegressor(random_state=42)
    grid_search = RandomizedSearchCV(
        gb_base, param_grid, n_iter=20, cv=3, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=1, random_state=42
    )

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    print("Running hyperparameter tuning (20 iterations)...")
    grid_search.fit(X_combined, y_combined)

    best_model = grid_search.best_estimator_

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate
    train_metrics = evaluate_regression(y_train, y_train_pred, "TRAINING - Tuned GB")
    test_metrics = evaluate_regression(y_test, y_test_pred, "TEST - Tuned GB")

    return best_model, grid_search, train_metrics, test_metrics


def train_with_mlflow_manual_logging(X_train, X_val, X_test, y_train, y_val, y_test,
                                     output_dir: str = 'Membangun_model/artifacts'):
    """
    Train models with MLflow manual logging (NOT autolog)
    Includes hyperparameter tuning for Skilled level (3 poin)
    
    Args:
        X_train, X_val, X_test: Feature sets
        y_train, y_val, y_test: Target sets
        output_dir: Directory to save models
    """
    
    os.makedirs(output_dir, exist_ok=True)

    # Set MLflow tracking
    mlflow.set_experiment("Amazon_Daily_Revenue_Forecasting_Tuning")
    
    print("\n" + "=" * 80)
    print("KRITERIA 2: MODEL BUILDING - SKILLED LEVEL (3 Poin)")
    print("MLflow Manual Logging with Hyperparameter Tuning")
    print("=" * 80 + "\n")
    
    results = {}
    
    # ========== RANDOM FOREST TUNING ==========
    with mlflow.start_run(run_name="rf_tuned"):
        rf_model, rf_grid, rf_train_metrics, rf_test_metrics = hyperparameter_tuning_random_forest(
            X_train, y_train, X_val, y_val, X_test, y_test, output_dir
        )

        # MANUAL LOGGING (instead of autolog)
        mlflow.log_params({
            'model_type': 'RandomForestRegressor',
            'n_estimators': int(getattr(rf_model, 'n_estimators', 0)),
            'max_depth': str(getattr(rf_model, 'max_depth', None))
        })

        # Log regression metrics
        mlflow.log_metrics({
            'train_mae': rf_train_metrics['mae'],
            'train_rmse': rf_train_metrics['rmse'],
            'train_r2': rf_train_metrics['r2'],
            'train_mape_pct': rf_train_metrics['mape_pct'],
            'test_mae': rf_test_metrics['mae'],
            'test_rmse': rf_test_metrics['rmse'],
            'test_r2': rf_test_metrics['r2'],
            'test_mape_pct': rf_test_metrics['mape_pct']
        })

        # Save model
        model_path = f'{output_dir}/rf_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        mlflow.log_artifact(model_path)

        results['rf_tuned'] = {
            'model': rf_model,
            'test_rmse': rf_test_metrics['rmse'],
            'metrics': {
                'train': rf_train_metrics,
                'test': rf_test_metrics
            }
        }
    
    # ========== GRADIENT BOOSTING TUNING ==========
    with mlflow.start_run(run_name="gb_tuned"):
        gb_model, gb_grid, gb_train_metrics, gb_test_metrics = hyperparameter_tuning_gradient_boosting(
            X_train, y_train, X_val, y_val, X_test, y_test, output_dir
        )

        # MANUAL LOGGING
        mlflow.log_params({
            'model_type': 'GradientBoostingRegressor',
            'n_estimators': int(getattr(gb_model, 'n_estimators', 0)),
            'learning_rate': float(getattr(gb_model, 'learning_rate', 0.0)),
        })

        mlflow.log_metrics({
            'train_mae': gb_train_metrics['mae'],
            'train_rmse': gb_train_metrics['rmse'],
            'train_r2': gb_train_metrics['r2'],
            'train_mape_pct': gb_train_metrics['mape_pct'],
            'test_mae': gb_test_metrics['mae'],
            'test_rmse': gb_test_metrics['rmse'],
            'test_r2': gb_test_metrics['r2'],
            'test_mape_pct': gb_test_metrics['mape_pct']
        })

        # Save model
        model_path = f'{output_dir}/gb_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(gb_model, f)
        mlflow.log_artifact(model_path)

        results['gb_tuned'] = {
            'model': gb_model,
            'test_rmse': gb_test_metrics['rmse'],
            'metrics': {
                'train': gb_train_metrics,
                'test': gb_test_metrics
            }
        }
    
    # ========== COMPARISON ==========
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - TUNED MODELS")
    print("=" * 80)
    
    best_model = None
    best_rmse = float('inf')
    best_name = ""

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test RMSE: {result['test_rmse']:.4f}")

        if result['test_rmse'] < best_rmse:
            best_rmse = result['test_rmse']
            best_model = result['model']
            best_name = model_name

    print(f"\n{'=' * 80}")
    print(f"Best Tuned Model: {best_name} (RMSE: {best_rmse:.4f})")
    print(f"{'=' * 80}\n")

    # Save best model
    best_model_path = f'{output_dir}/best_tuned_model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to: {best_model_path}")

    return best_model, results


def main():
    """Main execution function"""
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DAILY SALES FORECASTING DATA")
    print("=" * 80 + "\n")

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()

    # Train with hyperparameter tuning
    best_model, results = train_with_mlflow_manual_logging(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\nKriteria 2 - Skilled Level (3 Poin) Complete!")
    print("Features:")
    print("  ✓ Hyperparameter tuning dengan RandomizedSearchCV")
    print("  ✓ Manual logging di MLflow (tidak autolog)")
    print("  ✓ Metrik regresi: MAE, RMSE, R2, MAPE")
    print("  ✓ Comparison antara RandomForestRegressor dan GradientBoostingRegressor")
    print("\nMLflow Tracking UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == '__main__':
    main()
