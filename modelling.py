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
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import pickle
import os
from pathlib import Path


def load_preprocessed_data(daily_csv: str = 'Eksperimen_SML_DanangAgungRestuAji/preprocessing/cleaned_amazon_sales.csv', force_aggregation: str = None):
    """
    Load cleaned sales forecasting CSV and produce train/val/test splits.

    This function is more flexible than before:
    - Accepts files that already contain `Daily_Revenue`.
    - If `Date` is present and `Amount` per-order exists, aggregates to daily revenue.
    - If `Year`/`Month`/`Day` (or `DayOfMonth`) are present, constructs a date index.
    - If only `Year`/`Month` are present, aggregates by month (first-of-month).
    - If no time index is available, falls back to per-order modeling using `Amount` as target
      and performs a random split.

    Also automatically includes any numeric `_encoded` columns and numeric order fields
    as features.
    """
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
    loaded_path = None
    for p in candidates:
        try:
            if p.exists():
                # Try parsing Date if available
                try:
                    df = pd.read_csv(p, parse_dates=['Date'])
                except Exception:
                    df = pd.read_csv(p)
                loaded_path = p
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
        raise FileNotFoundError(
            "Sales CSV not found. Please run preprocessing to generate it or place it in the preprocessing folder."
        )

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # If force_aggregation == 'daily', prefer aggregating Amount -> Daily_Revenue
    if force_aggregation == 'daily':
        # ensure Date exists or can be constructed
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
            # Fall back to other constructors below
            pass

    # If Daily_Revenue already exists, use it directly
    if 'Daily_Revenue' in df.columns:
        time_indexed = 'Date' in df.columns
        if time_indexed:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').dropna()
            # remove zero-revenue days (likely padded empty dates)
            if 'Daily_Revenue' in df.columns:
                nonzero_before = len(df)
                df = df[df['Daily_Revenue'] != 0]
                nonzero_after = len(df)
                print(f"Removed zero-revenue rows: {nonzero_before - nonzero_after}")
            # select numeric and encoded features
            feature_cols = [c for c in df.columns if c.endswith('_encoded') or c in ['lag_1','lag_7','rolling_mean_7','rolling_std_7','day','month','year','weekday','weekofyear']]
            feature_cols = [c for c in feature_cols if c in df.columns]
            X = df[feature_cols]
            y = df['Daily_Revenue']
            n = len(df)
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

    # If Date present and Amount exists, aggregate to daily revenue
    if 'Date' in df.columns and 'Amount' in df.columns:
        tmp = df.copy()
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        daily = tmp.groupby('Date', as_index=True).agg({'Amount': 'sum'})
        daily = daily.rename(columns={'Amount': 'Daily_Revenue'})
        # remove zero-revenue days introduced by sparse data
        nonzero_before = len(daily)
        daily = daily[daily['Daily_Revenue'] != 0]
        nonzero_after = len(daily)
        print(f"Removed zero-revenue days after aggregation: {nonzero_before - nonzero_after}")

        # create simple time features
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

    # If Year/Month/(Day or DayOfMonth) available, construct Date and aggregate
    if 'Amount' in df.columns and 'Year' in df.columns and 'Month' in df.columns:
        tmp = df.copy()
        if 'Day' in tmp.columns or 'DayOfMonth' in tmp.columns:
            day_col = 'Day' if 'Day' in tmp.columns else 'DayOfMonth'
            try:
                tmp['Date'] = pd.to_datetime(tmp[['Year','Month',day_col]])
            except Exception:
                # fallback: set day to 1
                tmp['Date'] = pd.to_datetime(tmp['Year'].astype(int).astype(str) + '-' + tmp['Month'].astype(int).astype(str) + '-01')
        else:
            # no day-of-month, aggregate by month (first of month)
            tmp['Date'] = pd.to_datetime(tmp['Year'].astype(int).astype(str) + '-' + tmp['Month'].astype(int).astype(str) + '-01')

        daily = tmp.groupby('Date', as_index=True).agg({'Amount': 'sum'})
        daily = daily.rename(columns={'Amount': 'Daily_Revenue'})
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

        print(f"Constructed date from Year/Month and aggregated: total={n}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # Fallback: if no time index available but Amount exists, do per-order modeling
    if 'Amount' in df.columns:
        df2 = df.copy()
        # choose numeric/encoded features automatically
        feature_cols = [c for c in df2.columns if c.endswith('_encoded') or df2[c].dtype.kind in 'biufc' and c != 'Amount']
        if not feature_cols:
            # minimal features: Year, Month, DayOfWeek if present
            for c in ['Year','Month','DayOfWeek','Qty','B2B']:
                if c in df2.columns:
                    feature_cols.append(c)

        X = df2[feature_cols].fillna(0)
        y = df2['Amount']

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"Fallback per-order modeling using Amount: total={len(X)}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_val.reset_index(drop=True), y_test.reset_index(drop=True)

    raise ValueError("Unable to prepare features/target from the provided CSV. Ensure file contains 'Amount' and/or 'Date' or a precomputed 'Daily_Revenue'.")


def evaluate_regression(y_true, y_pred, dataset_name: str = ""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Stable MAPE calculation to avoid huge values when y_true has zeros/small values
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
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
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=8, min_samples_leaf=5),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, n_iter_no_change=10, validation_fraction=0.1, tol=1e-4)
    }

    best_model = None
    best_rmse = float('inf')
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        # Basic label / type safety and diagnostics
        y_train = pd.to_numeric(y_train, errors='coerce')
        y_val = pd.to_numeric(y_val, errors='coerce')
        y_test = pd.to_numeric(y_test, errors='coerce')

        def _stats(series):
            return {'n': len(series), 'mean': float(series.mean()), 'std': float(series.std()), 'min': float(series.min()), 'max': float(series.max())}

        tr_stats = _stats(y_train)
        va_stats = _stats(y_val)
        te_stats = _stats(y_test)

        print(f"Target stats - train: n={tr_stats['n']}, mean={tr_stats['mean']:.2f}, std={tr_stats['std']:.2f}")
        print(f"Target stats - val  : n={va_stats['n']}, mean={va_stats['mean']:.2f}, std={va_stats['std']:.2f}")
        print(f"Target stats - test : n={te_stats['n']}, mean={te_stats['mean']:.2f}, std={te_stats['std']:.2f}")

        # detect large distribution shift
        if tr_stats['mean'] != 0 and abs(tr_stats['mean'] - va_stats['mean']) / (abs(tr_stats['mean']) + 1e-9) > 0.5:
            print("WARNING: Large shift between train and validation target means (>50%). Check your splits or aggregation logic.")

        # Drop any accidental target columns from features
        def _clean_X(X):
            Xc = X.copy()
            for tcol in ['Daily_Revenue','Amount']:
                if tcol in Xc.columns:
                    print(f"Dropping target-like column from features: {tcol}")
                    Xc = Xc.drop(columns=[tcol])
            # Ensure numeric dtype for model
            Xc = Xc.select_dtypes(include=[np.number]).fillna(0)
            return Xc

        X_train_clean = _clean_X(X_train)
        X_val_clean = _clean_X(X_val)
        X_test_clean = _clean_X(X_test)

        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.autolog()

            # Fit using cleaned numeric-only features
            model.fit(X_train_clean, y_train)

            y_train_pred = model.predict(X_train_clean)
            # if validation set is empty (e.g., some artifact loads) guard predictions
            y_val_pred = model.predict(X_val_clean) if len(X_val_clean) > 0 else np.array([])
            y_test_pred = model.predict(X_test_clean) if len(X_test_clean) > 0 else np.array([])

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

            # Feature importance for tree models
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    fname = list(X_train_clean.columns)
                    fi = sorted(zip(fname, importances), key=lambda x: x[1], reverse=True)[:20]
                    print(f"Top feature importances for {model_name}:")
                    for f, v in fi:
                        print(f"  {f}: {v:.4f}")
            except Exception:
                pass

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
    
    # support running with FORCE_AGG environment variable or CLI arg --force-agg
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
    
    # Train models
    best_model, results = train_model_basic(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\nKriteria 2 - Basic Level (2 Poin) Complete!")
    print("MLflow Tracking UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == '__main__':
    main()
