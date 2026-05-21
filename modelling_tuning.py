"""
Model Training with Hyperparameter Tuning - Kriteria 2: Skilled Level (3 Poin)
Uses MLflow Tracking UI (local) with MANUAL logging (not autolog)
Includes hyperparameter tuning and additional metrics
Author: Danang Agung Restu Aji
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
try:
    import dagshub
except Exception:
    dagshub = None
import pickle
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_preprocessed_data(daily_csv: str = 'Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_sales_forecasting.csv', force_aggregation: str = None):
    """
    Flexible loader adapted to new preprocessing outputs (Amount, Year/Month, *_encoded).
    Produces X_train, X_val, X_test, y_train, y_val, y_test.
    """
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        _resolve(daily_csv),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_sales_ing.csv'),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/cleaned_amazon_sales.csv')
    ]

    df = None
    for p in candidates:
        try:
            if p.exists():
                try:
                    df = pd.read_csv(p, parse_dates=['Date'])
                except Exception:
                    df = pd.read_csv(p)
                print(f"Loaded CSV from: {p}")
                break
        except Exception:
            continue

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
        raise FileNotFoundError(f"Sales CSV not found. Tried: {[str(p) for p in candidates]}")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # If force_aggregation == 'daily', prefer aggregating Amount -> Daily_Revenue
    if force_aggregation == 'daily':
        if 'Date' in df.columns and 'Amount' in df.columns:
            tmp = df.copy()
            tmp['Date'] = pd.to_datetime(tmp['Date'])
            daily = tmp.groupby('Date', as_index=True).agg({'Amount': 'sum'}).rename(columns={'Amount':'Daily_Revenue'})
            # remove zero-revenue days introduced by sparse data
            nonzero_before = len(daily)
            daily = daily[daily['Daily_Revenue'] != 0]
            nonzero_after = len(daily)
            print(f"Removed zero-revenue days after aggregation: {nonzero_before - nonzero_after}")
            daily['lag_1'] = daily['Daily_Revenue'].shift(1)
            daily['lag_7'] = daily['Daily_Revenue'].shift(7)
            daily['rolling_mean_7'] = daily['Daily_Revenue'].rolling(window=7).mean()
            daily['rolling_std_7'] = daily['Daily_Revenue'].rolling(window=7).std()
            daily['day'] = daily.index.day
            daily['month'] = daily.index.month
            daily['year'] = daily.index.year
            daily['weekday'] = daily.index.weekday
            try:
                daily['weekofyear'] = daily.index.isocalendar().week
            except Exception:
                daily['weekofyear'] = daily.index.week
            daily = daily.dropna().reset_index()
            feature_cols = [c for c in ['lag_1','lag_7','rolling_mean_7','rolling_std_7','day','month','year','weekday','weekofyear'] if c in daily.columns]
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
            print(f"Force-aggregated daily revenue: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            pass

    # If Daily_Revenue present and Date exists, use it
    if 'Daily_Revenue' in df.columns and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna()
        # remove zero-revenue rows
        nonzero_before = len(df)
        df = df[df['Daily_Revenue'] != 0]
        nonzero_after = len(df)
        print(f"Removed zero-revenue rows: {nonzero_before - nonzero_after}")
        feature_cols = [c for c in df.columns if c.endswith('_encoded') or c in ['lag_1','lag_7','rolling_mean_7','rolling_std_7','day','month','year','weekday','weekofyear']]
        X = df[[c for c in feature_cols if c in df.columns]]
        y = df['Daily_Revenue']
        n = len(df)
        if n < 10:
            raise ValueError("After removing zero-revenue rows there are too few samples for modeling.")
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        X_train = X.iloc[:n_train].reset_index(drop=True)
        y_train = y.iloc[:n_train].reset_index(drop=True)
        X_val = X.iloc[n_train:n_train + n_val].reset_index(drop=True)
        y_val = y.iloc[n_train:n_train + n_val].reset_index(drop=True)
        X_test = X.iloc[n_train + n_val:].reset_index(drop=True)
        y_test = y.iloc[n_train + n_val:].reset_index(drop=True)

        print(f"Loaded Daily_Revenue dataset: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # If Date and Amount present, aggregate to daily revenue
    if 'Date' in df.columns and 'Amount' in df.columns:
        tmp = df.copy()
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        daily = tmp.groupby('Date', as_index=True).agg({'Amount': 'sum'}).rename(columns={'Amount':'Daily_Revenue'})
        # remove zero-revenue days introduced by sparse data
        nonzero_before = len(daily)
        daily = daily[daily['Daily_Revenue'] != 0]
        nonzero_after = len(daily)
        print(f"Removed zero-revenue days after aggregation: {nonzero_before - nonzero_after}")
        daily['lag_1'] = daily['Daily_Revenue'].shift(1)
        daily['lag_7'] = daily['Daily_Revenue'].shift(7)
        daily['rolling_mean_7'] = daily['Daily_Revenue'].rolling(window=7).mean()
        daily['rolling_std_7'] = daily['Daily_Revenue'].rolling(window=7).std()
        daily['day'] = daily.index.day
        daily['month'] = daily.index.month
        daily['year'] = daily.index.year
        daily['weekday'] = daily.index.weekday
        try:
            daily['weekofyear'] = daily.index.isocalendar().week
        except Exception:
            daily['weekofyear'] = daily.index.week
        daily = daily.dropna().reset_index()
        if len(daily) < 10:
            raise ValueError("After removing zero-revenue days there are too few samples for modeling. Consider different aggregation or keep zeros.")

        feature_cols = [c for c in ['lag_1','lag_7','rolling_mean_7','rolling_std_7','day','month','year','weekday','weekofyear'] if c in daily.columns]
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

        print(f"Aggregated daily revenue from Amount: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # If Year/Month present, try to construct Date then aggregate
    if 'Amount' in df.columns and 'Year' in df.columns and 'Month' in df.columns:
        tmp = df.copy()
        if 'Day' in tmp.columns:
            try:
                tmp['Date'] = pd.to_datetime(tmp[['Year','Month','Day']])
            except Exception:
                tmp['Date'] = pd.to_datetime(tmp['Year'].astype(int).astype(str) + '-' + tmp['Month'].astype(int).astype(str) + '-01')
        else:
            tmp['Date'] = pd.to_datetime(tmp['Year'].astype(int).astype(str) + '-' + tmp['Month'].astype(int).astype(str) + '-01')

        daily = tmp.groupby('Date', as_index=True).agg({'Amount': 'sum'}).rename(columns={'Amount':'Daily_Revenue'})
        # remove zero-revenue days introduced by sparse data
        nonzero_before = len(daily)
        daily = daily[daily['Daily_Revenue'] != 0]
        nonzero_after = len(daily)
        print(f"Removed zero-revenue days after aggregation: {nonzero_before - nonzero_after}")
        daily['lag_1'] = daily['Daily_Revenue'].shift(1)
        daily['lag_7'] = daily['Daily_Revenue'].shift(7)
        daily['rolling_mean_7'] = daily['Daily_Revenue'].rolling(window=7).mean()
        daily['rolling_std_7'] = daily['Daily_Revenue'].rolling(window=7).std()
        daily['day'] = daily.index.day
        daily['month'] = daily.index.month
        daily['year'] = daily.index.year
        daily['weekday'] = daily.index.weekday
        try:
            daily['weekofyear'] = daily.index.isocalendar().week
        except Exception:
            daily['weekofyear'] = daily.index.week
        daily = daily.dropna().reset_index()

        feature_cols = [c for c in ['lag_1','lag_7','rolling_mean_7','rolling_std_7','day','month','year','weekday','weekofyear'] if c in daily.columns]
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

        print(f"Constructed date from Year/Month and aggregated: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # Fallback: per-order modeling using Amount (random split)
    if 'Amount' in df.columns:
        df2 = df.copy()
        # automatic numeric feature selection (include encoded)
        feature_cols = [c for c in df2.columns if c.endswith('_encoded') or df2[c].dtype.kind in 'biufc' and c != 'Amount']
        if not feature_cols:
            for c in ['Year','Month','DayOfWeek','Qty','B2B']:
                if c in df2.columns:
                    feature_cols.append(c)
        X = df2[feature_cols].fillna(0)
        y = df2['Amount']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"Fallback per-order modeling using Amount: total={len(X)}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True)

    raise ValueError("Unable to prepare features/target from provided CSV. Ensure it contains 'Amount'/'Date' or precomputed 'Daily_Revenue'.")


def _clean_X_for_model(X):
    """Return numeric-only dataframe and drop any target-like columns."""
    Xc = X.copy()
    for tcol in ['Daily_Revenue', 'Amount']:
        if tcol in Xc.columns:
            Xc = Xc.drop(columns=[tcol])
    # keep numeric types only
    Xc = Xc.select_dtypes(include=[np.number]).fillna(0)
    return Xc


def evaluate_regression(y_true, y_pred, dataset_name: str = ""):
    """
    Evaluate regression predictions with common metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Stable MAPE - avoid huge values when y_true is zero or extremely small
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    # set eps to a small fraction of typical scale (median abs) but at least 1e-3
    scale = np.median(np.abs(y_true_arr))
    eps = max(1e-3, 0.01 * scale)
    denom = np.where(np.abs(y_true_arr) < eps, eps, np.abs(y_true_arr))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = (np.abs((y_true_arr - y_pred_arr) / denom)).mean() * 100

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
        'max_features': ['sqrt', 'log2', None]
    }

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = RandomizedSearchCV(
        rf_base, param_grid, n_iter=20, cv=3, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=1, random_state=42
    )

    # Clean features (drop target-like columns, numeric-only)
    X_train_c = _clean_X_for_model(X_train)
    X_val_c = _clean_X_for_model(X_val)
    X_test_c = _clean_X_for_model(X_test)

    X_combined = pd.concat([X_train_c, X_val_c])
    y_combined = pd.concat([y_train, y_val])

    # Ensure target numeric
    y_combined = pd.to_numeric(y_combined, errors='coerce')

    print("Running hyperparameter tuning (20 iterations)...")
    grid_search.fit(X_combined, y_combined)

    best_model = grid_search.best_estimator_

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

    # Predictions
    y_train_pred = best_model.predict(X_train_c)
    y_test_pred = best_model.predict(X_test_c)

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

    X_train_c = _clean_X_for_model(X_train)
    X_val_c = _clean_X_for_model(X_val)
    X_test_c = _clean_X_for_model(X_test)

    X_combined = pd.concat([X_train_c, X_val_c])
    y_combined = pd.concat([y_train, y_val])
    y_combined = pd.to_numeric(y_combined, errors='coerce')

    print("Running hyperparameter tuning (20 iterations)...")
    grid_search.fit(X_combined, y_combined)

    best_model = grid_search.best_estimator_

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

    # Predictions
    y_train_pred = best_model.predict(X_train_c)
    y_test_pred = best_model.predict(X_test_c)

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

    # Inisialisasi Dagshub untuk MLflow tracking
    try:
        if dagshub is not None:
            dagshub.init(repo_owner='ProfDARA', repo_name='membangun_model', mlflow=True)
            print('dagshub.init called — MLflow will log to DagsHub')
    except Exception as e:
        print(f"dagshub.init skipped: {e}")

    # Set MLflow tracking URI to DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/ProfDARA/membangun_model.mlflow")
    mlflow.set_experiment("Amazon_Daily_Revenue_Forecasting_Tuning")
    
    print("\n" + "=" * 80)
    print("KRITERIA 2: MODEL BUILDING - SKILLED LEVEL (3 Poin)")
    print("MLflow Manual Logging with Hyperparameter Tuning")
    print("=" * 80 + "\n")
    
    results = {}

    # Basic diagnostics on target distributions and sample data
    def _stats(s):
        s = pd.to_numeric(s, errors='coerce')
        return {'n': len(s), 'mean': float(s.mean()), 'std': float(s.std()), 'min': float(np.min(s)), 'max': float(np.max(s))}

    tr = _stats(y_train)
    va = _stats(y_val)
    te = _stats(y_test)
    print(f"Target stats - train mean={tr['mean']:.2f}, val mean={va['mean']:.2f}, test mean={te['mean']:.2f}")
    if tr['mean'] != 0 and (abs(tr['mean'] - va['mean']) / (abs(tr['mean']) + 1e-9) > 0.5 or abs(tr['mean'] - te['mean']) / (abs(tr['mean']) + 1e-9) > 0.5):
        print("WARNING: Large shift between train and validation/test target means (>50%). Check whether splits mix per-order and aggregated (daily/monthly) targets.")

    # show small samples and dtypes for quick inspection
    try:
        print("X_train sample dtypes:\n", X_train.dtypes.head(10))
        print("X_train sample rows:\n", X_train.head(3).to_dict(orient='records'))
        print("y_train sample:\n", y_train.head(5).tolist())
        print("X_test sample rows:\n", X_test.head(3).to_dict(orient='records'))
        print("y_test sample:\n", y_test.head(5).tolist())
    except Exception:
        pass
    
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

        # Log tuning results (best params + compact cv summary)
        try:
            best_params = rf_grid.best_params_
            mlflow.log_params({'rf_best_'+k: str(v) for k, v in best_params.items()})
            # Create compact CV results summary
            cv_results = rf_grid.cv_results_
            # extract top 3 by rank_test_score
            top_idx = sorted(range(len(cv_results['rank_test_score'])), key=lambda i: cv_results['rank_test_score'][i])[:3]
            cv_summary = []
            for i in top_idx:
                cv_summary.append({
                    'params': cv_results['params'][i],
                    'mean_test_score': float(cv_results['mean_test_score'][i]),
                    'std_test_score': float(cv_results['std_test_score'][i]),
                    'rank_test_score': int(cv_results['rank_test_score'][i])
                })
            cv_summary_path = Path(output_dir) / 'rf_cv_summary.json'
            with open(cv_summary_path, 'w') as jf:
                json.dump(cv_summary, jf, indent=2)
            mlflow.log_artifact(str(cv_summary_path))
        except Exception:
            pass

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

        # Additional artefak untuk Skilled: feature importances + residuals
        try:
            importances = getattr(rf_model, 'feature_importances_', None)
            if importances is not None:
                feat_names = list(_clean_X_for_model(X_train).columns)
                fig, ax = plt.subplots(figsize=(8, max(3, len(feat_names) * 0.3)))
                sns.barplot(x=importances, y=feat_names, ax=ax)
                ax.set_title('RF Feature Importances')
                fig.tight_layout()
                fi_path = Path(output_dir) / 'rf_feature_importances.png'
                fig.savefig(fi_path)
                plt.close(fig)
                mlflow.log_artifact(str(fi_path))
        except Exception:
            pass

        try:
            y_test_pred = rf_model.predict(X_test)
            residuals = y_test - y_test_pred
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Residual (True - Pred)')
            ax.set_title('RF Residuals vs Predicted')
            fig.tight_layout()
            resid_path = Path(output_dir) / 'rf_residuals.png'
            fig.savefig(resid_path)
            plt.close(fig)
            mlflow.log_artifact(str(resid_path))
        except Exception:
            pass

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

        # Log GB tuning results and cv summary
        try:
            best_params = gb_grid.best_params_
            mlflow.log_params({'gb_best_'+k: str(v) for k, v in best_params.items()})
            cv_results = gb_grid.cv_results_
            top_idx = sorted(range(len(cv_results['rank_test_score'])), key=lambda i: cv_results['rank_test_score'][i])[:3]
            cv_summary = []
            for i in top_idx:
                cv_summary.append({
                    'params': cv_results['params'][i],
                    'mean_test_score': float(cv_results['mean_test_score'][i]),
                    'std_test_score': float(cv_results['std_test_score'][i]),
                    'rank_test_score': int(cv_results['rank_test_score'][i])
                })
            cv_summary_path = Path(output_dir) / 'gb_cv_summary.json'
            with open(cv_summary_path, 'w') as jf:
                json.dump(cv_summary, jf, indent=2)
            mlflow.log_artifact(str(cv_summary_path))
        except Exception:
            pass

        # Save model
        model_path = f'{output_dir}/gb_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(gb_model, f)
        mlflow.log_artifact(model_path)

        # Additional artefak: feature importances + residuals for GB
        try:
            importances = getattr(gb_model, 'feature_importances_', None)
            if importances is not None:
                feat_names = list(_clean_X_for_model(X_train).columns)
                fig, ax = plt.subplots(figsize=(8, max(3, len(feat_names) * 0.3)))
                sns.barplot(x=importances, y=feat_names, ax=ax)
                ax.set_title('GB Feature Importances')
                fig.tight_layout()
                fi_path = Path(output_dir) / 'gb_feature_importances.png'
                fig.savefig(fi_path)
                plt.close(fig)
                mlflow.log_artifact(str(fi_path))
        except Exception:
            pass

        try:
            y_test_pred = gb_model.predict(X_test)
            residuals = y_test - y_test_pred
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Residual (True - Pred)')
            ax.set_title('GB Residuals vs Predicted')
            fig.tight_layout()
            resid_path = Path(output_dir) / 'gb_residuals.png'
            fig.savefig(resid_path)
            plt.close(fig)
            mlflow.log_artifact(str(resid_path))
        except Exception:
            pass

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

    # allow forcing aggregation via env var or CLI
    import os, sys
    force_agg = None
    if os.environ.get('FORCE_AGG', '').lower() == 'daily':
        force_agg = 'daily'
    for i, a in enumerate(sys.argv[1:]):
        if a in ('--force-agg',) and i + 2 <= len(sys.argv[1:]):
            val = sys.argv[1:][i+1]
            if val == 'daily':
                force_agg = 'daily'

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(force_aggregation=force_agg)

    # Train with hyperparameter tuning
    best_model, results = train_with_mlflow_manual_logging(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\nKriteria 2 - Skilled Level (3 Poin) Complete!")
    print("Features:")
    print("  - Hyperparameter tuning dengan RandomizedSearchCV")
    print("  - Manual logging di MLflow (tidak autolog)")
    print("  - Metrik regresi: MAE, RMSE, R2, MAPE")
    print("  - Comparison antara RandomForestRegressor dan GradientBoostingRegressor")
    print("\nMLflow Tracking UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == '__main__':
    main()
