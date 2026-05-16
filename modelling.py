"""
Model Training - Kriteria 2: Basic Level
Uses MLflow Tracking UI (local) with autolog
Author: Danang Agung Restu Aji
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import os
from pathlib import Path


def load_preprocessed_data(daily_csv: str = 'Eksperimen_SML_DanangAgungRestuAji/preprocessing/cleaned_amazon_sales.csv'):
    """
    Load cleaned daily sales forecasting CSV and perform time-based split.

    Tries multiple fallback locations and supports loading pre-split artifacts
    if present. If file(s) are missing, raises an informative error with next steps.
    """
    # Resolve candidate paths relative to the repository root so scripts work
    # whether run from repo root or from inside subfolders like Membangun_model.
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        _resolve(daily_csv),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/cleaned_amazon_sales.csv'),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_sales_forecasting.csv')
    ]

    df = None
    for p in candidates:
        try:
            if p.exists():
                df = pd.read_csv(p, parse_dates=['Date'])
                print(f"Loaded daily CSV from: {p}")
                break
        except Exception:
            try:
                df = pd.read_csv(p)
                print(f"Loaded daily CSV (no Date parse) from: {p}")
                break
            except Exception:
                df = None

    artifacts_dir = repo_root / 'Eksperimen_SML_DanangAgungRestuAji' / 'preprocessing' / 'preprocessing_artifacts'
    if df is None and artifacts_dir.exists():
        x_train_path = artifacts_dir / 'X_train.csv'
        y_train_path = artifacts_dir / 'y_train.csv'
        if x_train_path.exists() and y_train_path.exists():
            X_train = pd.read_csv(x_train_path)
            X_val = pd.read_csv(artifacts_dir / 'X_val.csv') if (artifacts_dir / 'X_val.csv').exists() else pd.DataFrame()
            X_test = pd.read_csv(artifacts_dir / 'X_test.csv') if (artifacts_dir / 'X_test.csv').exists() else pd.DataFrame()
            y_train = pd.read_csv(y_train_path).squeeze()
            y_val = pd.read_csv(artifacts_dir / 'y_val.csv').squeeze() if (artifacts_dir / 'y_val.csv').exists() else pd.Series(dtype=float)
            y_test = pd.read_csv(artifacts_dir / 'y_test.csv').squeeze() if (artifacts_dir / 'y_test.csv').exists() else pd.Series(dtype=float)

            print(f"Loaded pre-split artifacts from: {artifacts_dir}")
            return X_train, X_val, X_test, y_train, y_val, y_test

    if df is None:
        raise FileNotFoundError(
            "daily_sales_forecasting.csv not found.\n"
            "Please run preprocessing to generate it:\n"
            "  cd Eksperimen_SML_DanangAgungRestuAji/preprocessing\n"
            "  python automate_DanangAReAji.py\n"
            "Or ensure the file exists at one of: \n"
            f"  {[str(p) for p in candidates]}\n"
        )

    expected_features = [
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7',
        'day', 'month', 'year', 'weekday', 'weekofyear'
    ]

    available = [c for c in expected_features if c in df.columns]
    if len(available) < 3:
        # Attempt to build time-series features from cleaned raw data if Date and Amount/Revenue exist
        try:
            if 'Date' in df.columns:
                # Aggregate to daily revenue if raw sales exist
                if 'Amount' in df.columns or 'Daily_Revenue' not in df.columns:
                    # prefer Daily_Revenue if present, otherwise sum Amount per date
                    if 'Daily_Revenue' in df.columns:
                        daily = df.copy()
                        daily['Date'] = pd.to_datetime(daily['Date'])
                        daily = daily.sort_values('Date')
                        daily = daily.set_index('Date')
                    else:
                        tmp = df.copy()
                        tmp['Date'] = pd.to_datetime(tmp['Date'])
                        daily = tmp.groupby('Date', as_index=True).agg({'Amount': 'sum'})
                        daily = daily.rename(columns={'Amount': 'Daily_Revenue'})

                    # Ensure index is daily and continuous (fill missing dates)
                    daily = daily.asfreq('D', fill_value=0)

                    # Create features
                    daily['lag_1'] = daily['Daily_Revenue'].shift(1)
                    daily['lag_7'] = daily['Daily_Revenue'].shift(7)
                    daily['rolling_mean_7'] = daily['Daily_Revenue'].rolling(window=7).mean()
                    daily['rolling_std_7'] = daily['Daily_Revenue'].rolling(window=7).std()
                    daily['day'] = daily.index.day
                    daily['month'] = daily.index.month
                    daily['year'] = daily.index.year
                    daily['weekday'] = daily.index.weekday
                    # weekofyear for pandas >=1.1 uses isocalendar
                    try:
                        daily['weekofyear'] = daily.index.isocalendar().week
                    except Exception:
                        daily['weekofyear'] = daily.index.week

                    daily = daily.dropna().reset_index()

                    # prepare X and y
                    feature_cols = [c for c in expected_features if c in daily.columns]
                    if len(feature_cols) < 3:
                        raise ValueError(f"Feature engineering produced insufficient features. Available: {feature_cols}")

                    X = daily[feature_cols]
                    y = daily['Daily_Revenue']

                    n = len(daily)
                    n_train = int(n * 0.7)
                    n_val = int(n * 0.15)

                    X_train = X.iloc[:n_train].reset_index(drop=True)
                    y_train = y.iloc[:n_train].reset_index(drop=True)

                    X_val = X.iloc[n_train:n_train + n_val].reset_index(drop=True)
                    y_val = y.iloc[n_train:n_train + n_val].reset_index(drop=True)

                    X_test = X.iloc[n_train + n_val:].reset_index(drop=True)
                    y_test = y.iloc[n_train + n_val:].reset_index(drop=True)

                    print(f"Engineered features from cleaned data: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
                    return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            raise ValueError(f"Not enough feature columns found in {p}. Available: {available}. Feature engineering failed: {e}")

    df = df.dropna().sort_values('Date') if 'Date' in df.columns else df.dropna()

    feature_cols = available
    target_col = 'Daily_Revenue'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {p}")

    X = df[feature_cols]
    y = df[target_col]

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
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
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


def train_model_basic(X_train, X_val, X_test, y_train, y_val, y_test,
                      output_dir: str = 'Membangun_model/artifacts'):
    os.makedirs(output_dir, exist_ok=True)

    # Set MLflow tracking
    mlflow.set_experiment("Amazon_Daily_Revenue_Forecasting")

    print("\n" + "=" * 80)
    print("KRITERIA 2: MODEL BUILDING - BASIC LEVEL (MLflow Autolog) - REGRESSION")
    print("=" * 80 + "\n")

    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_rmse = float('inf')
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.autolog()

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            train_metrics = evaluate_regression(y_train, y_train_pred, f"TRAINING - {model_name}")
            val_metrics = evaluate_regression(y_val, y_val_pred, f"VALIDATION - {model_name}")
            test_metrics = evaluate_regression(y_test, y_test_pred, f"TEST - {model_name}")

            mlflow.log_metrics({
                'train_mae': train_metrics['mae'],
                'train_rmse': train_metrics['rmse'],
                'train_r2': train_metrics['r2'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2']
            })

            mlflow.log_params({'model_type': model_name, 'train_size': len(X_train)})

            model_path = f'{output_dir}/{model_name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)

            results[model_name] = {
                'model': model,
                'test_rmse': test_metrics['rmse'],
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }

            if test_metrics['rmse'] < best_rmse:
                best_rmse = test_metrics['rmse']
                best_model = model
                best_model_name = model_name

    print("\n" + "=" * 80)
    print("MODEL TRAINING SUMMARY")
    print("=" * 80)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test RMSE: {result['test_rmse']:.4f}")

    print(f"\nBest Model: {best_model_name} (RMSE: {best_rmse:.4f})")
    print("=" * 80 + "\n")

    best_model_path = f'{output_dir}/best_model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to: {best_model_path}")

    return best_model, results


def main():
    """Main execution function"""
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80 + "\n")
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    # Train models
    best_model, results = train_model_basic(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\nKriteria 2 - Basic Level (2 Poin) Complete!")
    print("MLflow Tracking UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == '__main__':
    main()
