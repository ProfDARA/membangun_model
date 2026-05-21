"""
Demand Forecasting - Kriteria 2: Basic Level
Grouped daily demand (Category/State/SKU) with MLflow autolog
Author: Danang Agung Restu Aji
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _resolve_csv_path(csv_path: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]

    def _resolve(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (repo_root / p)

    candidates = [
        _resolve(csv_path),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/cleaned_amazon_sales.csv'),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_demand_forecasting.csv'),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_demand_by_sku.csv'),
        _resolve('Eksperimen_SML_DanangAgungRestuAji/preprocessing/daily_demand_by_state.csv')
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Sales CSV not found. Please run preprocessing to generate it or place it in the preprocessing folder."
    )


def load_sales_data(csv_path: str) -> pd.DataFrame:
    path = _resolve_csv_path(csv_path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path}") from exc

    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded CSV from: {path}")
    return df


def _normalize_group_value(value: str) -> str:
    return str(value).strip().lower()


def build_grouped_demand_dataset(
    df: pd.DataFrame,
    group_col: str,
    group_value: Optional[str] = None,
    date_col: str = 'Date',
    target_col: str = 'Qty',
    min_group_size: int = 30,
    fill_missing_dates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    work = df[[date_col, group_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
    work[group_col] = work[group_col].astype(str).str.strip()
    work = work.dropna(subset=[date_col, group_col, target_col])

    if group_value:
        target_norm = _normalize_group_value(group_value)
        group_norm = work[group_col].str.strip().str.lower()
        work = work[group_norm == target_norm]
        if work.empty:
            sample_groups = (
                df[group_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()[:10]
            )
            raise ValueError(
                f"No rows found for {group_col}='{group_value}'. Sample values: {sample_groups}"
            )

    daily = (
        work.groupby([date_col, group_col], as_index=False)[target_col]
        .sum()
        .rename(columns={target_col: 'Daily_Demand'})
    )

    if fill_missing_dates:
        frames = []
        for g, gdf in daily.groupby(group_col):
            full_idx = pd.date_range(gdf[date_col].min(), gdf[date_col].max(), freq='D')
            gdf = gdf.set_index(date_col).reindex(full_idx)
            gdf.index.name = date_col
            gdf[group_col] = g
            gdf['Daily_Demand'] = gdf['Daily_Demand'].fillna(0)
            frames.append(gdf.reset_index())
        daily = pd.concat(frames, ignore_index=True)

    if not group_value:
        group_counts = daily.groupby(group_col).size().sort_values(ascending=False)
        keep_groups = group_counts[group_counts >= min_group_size].index
        daily = daily[daily[group_col].isin(keep_groups)]

    if daily.empty:
        raise ValueError("No rows available after aggregation. Check group filters or min_group_size.")

    daily = daily.sort_values([group_col, date_col])
    daily['lag_1'] = daily.groupby(group_col)['Daily_Demand'].shift(1)
    daily['lag_7'] = daily.groupby(group_col)['Daily_Demand'].shift(7)
    daily['lag_14'] = daily.groupby(group_col)['Daily_Demand'].shift(14)
    daily['rolling_mean_7'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=7).mean()
    )
    daily['rolling_std_7'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=7).std()
    )
    daily['rolling_mean_30'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=30).mean()
    )
    daily['rolling_std_30'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=30).std()
    )

    daily['day'] = daily[date_col].dt.day
    daily['month'] = daily[date_col].dt.month
    daily['year'] = daily[date_col].dt.year
    daily['weekday'] = daily[date_col].dt.weekday
    daily['weekofyear'] = daily[date_col].dt.isocalendar().week.astype(int)
    daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)

    daily = daily.dropna().reset_index(drop=True)

    cat = daily[group_col].astype('category')
    daily['group_id'] = cat.cat.codes
    group_mapping = {str(name): int(idx) for idx, name in enumerate(cat.cat.categories)}

    return daily, group_mapping


def time_based_split(
    df: pd.DataFrame,
    date_col: str = 'Date',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df[date_col].unique())
    if len(unique_dates) < 3:
        raise ValueError("Not enough unique dates for time-based split.")

    n_dates = len(unique_dates)
    train_cut = max(1, int(n_dates * train_ratio))
    val_cut = max(train_cut + 1, int(n_dates * (train_ratio + val_ratio)))

    train_end = unique_dates[train_cut - 1]
    val_end = unique_dates[min(val_cut - 1, n_dates - 1)]

    train_df = df[df[date_col] <= train_end].copy()
    val_df = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test_df = df[df[date_col] > val_end].copy()

    return train_df, val_df, test_df


def prepare_train_data(
    csv_path: str,
    group_col: str,
    group_value: Optional[str] = None,
    date_col: str = 'Date',
    target_col: str = 'Qty',
    min_group_size: int = 30,
    fill_missing_dates: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, int]]:
    df = load_sales_data(csv_path)
    dataset, group_mapping = build_grouped_demand_dataset(
        df,
        group_col=group_col,
        group_value=group_value,
        date_col=date_col,
        target_col=target_col,
        min_group_size=min_group_size,
        fill_missing_dates=fill_missing_dates
    )

    train_df, val_df, test_df = time_based_split(dataset, date_col=date_col)
    include_group = dataset[group_col].nunique() > 1
    feature_cols = [
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_30', 'rolling_std_30',
        'day', 'month', 'year', 'weekday', 'weekofyear', 'is_weekend'
    ]
    if include_group:
        feature_cols.append('group_id')

    X_train = train_df[feature_cols]
    y_train = train_df['Daily_Demand']
    X_val = val_df[feature_cols]
    y_val = val_df['Daily_Demand']
    X_test = test_df[feature_cols]
    y_test = test_df['Daily_Demand']

    print(
        f"Prepared demand dataset: total={len(dataset)}, train={len(X_train)}, "
        f"val={len(X_val)}, test={len(X_test)}, groups={dataset[group_col].nunique()}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, group_mapping


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


def train_model_basic(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    output_dir: str = 'Membangun_model/artifacts',
    metadata: Optional[Dict[str, object]] = None
):
    os.makedirs(output_dir, exist_ok=True)

    # Set MLflow tracking
    group_col = metadata.get('group_col') if metadata else 'Group'
    mlflow.set_experiment(f"Amazon_Demand_Forecasting_{group_col}")

    print("\n" + "=" * 80)
    print("KRITERIA 2: MODEL BUILDING - BASIC LEVEL (MLflow Autolog) - DEMAND FORECASTING")
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
            for tcol in ['Daily_Revenue', 'Daily_Demand', 'Amount']:
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

            params = {
                'model_type': model_name,
                'train_size': len(X_train),
                'group_col': metadata.get('group_col') if metadata else None,
                'group_value': metadata.get('group_value') if metadata else None,
                'target_name': metadata.get('target_name') if metadata else None,
                'n_groups': metadata.get('n_groups') if metadata else None
            }
            params = {k: (str(v) if v is not None else 'None') for k, v in params.items()}
            mlflow.log_params(params)

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

    print("\n" + "=" * 80)
    print("LOADING DEMAND DATA")
    print("=" * 80 + "\n")

    import sys

    group_col = os.environ.get('GROUP_COL', 'Category')
    group_value = os.environ.get('GROUP_VALUE')
    target_col = os.environ.get('TARGET_COL', 'Qty')
    min_group_size = int(os.environ.get('MIN_GROUP_SIZE', '30'))
    csv_path = os.environ.get(
        'DEMAND_CSV',
        'Eksperimen_SML_DanangAgungRestuAji/preprocessing/cleaned_amazon_sales.csv'
    )

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--group-col' and i + 1 < len(args):
            group_col = args[i + 1]
        if a == '--group-value' and i + 1 < len(args):
            group_value = args[i + 1]
        if a == '--target-col' and i + 1 < len(args):
            target_col = args[i + 1]
        if a == '--min-group-size' and i + 1 < len(args):
            min_group_size = int(args[i + 1])
        if a == '--data-csv' and i + 1 < len(args):
            csv_path = args[i + 1]

    print(f"Group column: {group_col}")
    print(f"Group value : {group_value if group_value else 'ALL'}")
    print(f"Target col  : {target_col}")
    print(f"CSV path    : {csv_path}")

    X_train, X_val, X_test, y_train, y_val, y_test, group_map = prepare_train_data(
        csv_path=csv_path,
        group_col=group_col,
        group_value=group_value,
        target_col=target_col,
        min_group_size=min_group_size
    )

    meta = {
        'group_col': group_col,
        'group_value': group_value,
        'target_name': 'Daily_Demand',
        'n_groups': len(group_map)
    }

    best_model, results = train_model_basic(
        X_train, X_val, X_test, y_train, y_val, y_test, metadata=meta
    )

    print("\nKriteria 2 - Basic Level (2 Poin) Complete!")
    print("MLflow Tracking UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == '__main__':
    main()
