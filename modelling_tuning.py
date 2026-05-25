"""
Demand Forecasting with Hyperparameter Tuning
Uses MLflow Tracking UI (local) with MANUAL logging (not autolog)
Includes hyperparameter tuning and additional metrics
Author: Danang Agung Restu Aji
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
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
from typing import cast, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def _feature_columns_for_frame(df: pd.DataFrame) -> list[str]:
    base_features = [
        'category_encoded',
        'lag_1', 'lag_7',
        'rolling_mean_7', 'rolling_std_7',
        'day', 'month', 'weekday', 'is_weekend'
    ]
    feature_cols = [col for col in base_features if col in df.columns]
    return feature_cols


def _prepare_model_pair(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    Xc = _clean_X_for_model(X).reset_index(drop=True)
    yc = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors='coerce')
    valid_mask = yc.notna()
    Xc = Xc.loc[valid_mask].reset_index(drop=True)
    yc = yc.loc[valid_mask].reset_index(drop=True)
    return Xc, yc


def _baseline_naive_prediction(X: pd.DataFrame) -> pd.Series:
    if 'lag_1' not in X.columns:
        raise KeyError('lag_1 feature not found for naive baseline prediction')
    return pd.to_numeric(X['lag_1'], errors='coerce').fillna(0).reset_index(drop=True)


def _export_feature_importance(model, feature_names, output_dir: str, prefix: str):
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        return None

    importance_df = pd.DataFrame({
        'feature': list(feature_names)[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    out_path = Path(output_dir) / f'{prefix}_feature_importance.csv'
    importance_df.to_csv(out_path, index=False)
    return out_path


def _resolve_csv_path(csv_path: str) -> Path:
    repo_root = Path(__file__).resolve().parent

    def _resolve(p: str) -> Path:
        candidate = Path(p)
        return candidate if candidate.is_absolute() else (repo_root / candidate)

    candidates = [
        _resolve(csv_path),
        _resolve('amazon_preprocessing/daily_demand_forecasting.csv'),
        _resolve('amazon_preprocessing/daily_demand_by_sku.csv'),
        _resolve('amazon_preprocessing/daily_demand_by_state.csv'),
        _resolve('daily_demand_forecasting.csv'),
        _resolve('daily_demand_by_sku.csv'),
        _resolve('daily_demand_by_state.csv')
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
    work[target_col] = pd.to_numeric(work[target_col], errors='coerce')
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
    )
    daily = cast(pd.DataFrame, daily)
    daily.columns = [date_col, group_col, 'Daily_Demand']

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

    daily = cast(pd.DataFrame, daily.sort_values([group_col, date_col]))
    daily['lag_1'] = daily.groupby(group_col)['Daily_Demand'].shift(1)
    daily['lag_7'] = daily.groupby(group_col)['Daily_Demand'].shift(7)
    daily['rolling_mean_7'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=7).mean()
    )
    daily['rolling_std_7'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=7).std()
    )

    daily['day'] = daily[date_col].dt.day
    daily['month'] = daily[date_col].dt.month
    daily['weekday'] = daily[date_col].dt.weekday
    daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)

    daily = daily.dropna().reset_index(drop=True)

    cat = daily[group_col].astype('category')
    daily['category_encoded'] = cat.cat.codes
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
    demand_target_col = 'Daily_Demand' if 'Daily_Demand' in df.columns else target_col
    if demand_target_col not in df.columns:
        if 'Qty' in df.columns:
            demand_target_col = 'Qty'
        else:
            raise ValueError(
                f"Missing target column: expected 'Daily_Demand', '{target_col}', or 'Qty'."
            )

    dataset, group_mapping = build_grouped_demand_dataset(
        df,
        group_col=group_col,
        group_value=group_value,
        date_col=date_col,
        target_col=demand_target_col,
        min_group_size=min_group_size,
        fill_missing_dates=fill_missing_dates
    )
    target_column = 'Daily_Demand'

    train_df, val_df, test_df = time_based_split(dataset, date_col=date_col)
    feature_cols = _feature_columns_for_frame(dataset)

    X_train = train_df[feature_cols]
    y_train = train_df[target_column]
    X_val = val_df[feature_cols]
    y_val = val_df[target_column]
    X_test = test_df[feature_cols]
    y_test = test_df[target_column]

    print(
        f"Prepared demand dataset: total={len(dataset)}, train={len(X_train)}, "
        f"val={len(X_val)}, test={len(X_test)}, groups={dataset[group_col].nunique()}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, group_mapping


def encode_categorical_columns(df):
    df = df.copy()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def _clean_X_for_model(X):
    """Encode object columns, drop target-like columns, and return a numeric dataframe."""
    Xc = X.copy()
    for tcol in ['Daily_Revenue', 'Daily_Demand', 'Amount']:
        if tcol in Xc.columns:
            Xc = Xc.drop(columns=[tcol])
    for col in Xc.select_dtypes(include=['object']).columns:
        Xc[col] = LabelEncoder().fit_transform(Xc[col].astype(str))
    Xc = Xc.fillna(0)
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
    X_train_c, y_train_c = _prepare_model_pair(X_train, y_train)
    X_val_c, y_val_c = _prepare_model_pair(X_val, y_val)
    X_test_c, y_test_c = _prepare_model_pair(X_test, y_test)

    cv_splits = min(3, len(X_train_c) - 1)
    if cv_splits < 2:
        print("Not enough rows for cross-validation; fitting Random Forest with default parameters instead.")
        best_model = rf_base.fit(X_train_c, y_train_c)
        y_train_pred = best_model.predict(X_train_c) if len(X_train_c) > 0 else np.array([])
        y_val_pred = best_model.predict(X_val_c) if len(X_val_c) > 0 else np.array([])
        y_test_pred = best_model.predict(X_test_c) if len(X_test_c) > 0 else np.array([])
        train_metrics = evaluate_regression(y_train_c, y_train_pred, "TRAINING - Tuned RF")
        val_metrics = evaluate_regression(y_val_c, y_val_pred, "VALIDATION - Tuned RF")
        test_metrics = evaluate_regression(y_test_c, y_test_pred, "TEST - Tuned RF")
        return best_model, None, train_metrics, val_metrics, test_metrics

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    grid_search = RandomizedSearchCV(
        rf_base, param_grid, n_iter=20, cv=tscv, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=1, random_state=42
    )

    print("Running hyperparameter tuning (20 iterations)...")
    grid_search.fit(X_train_c, y_train_c)

    best_model = cast(RandomForestRegressor, grid_search.best_estimator_)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

    # Predictions
    y_train_pred = best_model.predict(X_train_c)
    y_val_pred = best_model.predict(X_val_c)
    y_test_pred = best_model.predict(X_test_c)

    # Evaluate
    train_metrics = evaluate_regression(y_train_c, y_train_pred, "TRAINING - Tuned RF")
    val_metrics = evaluate_regression(y_val_c, y_val_pred, "VALIDATION - Tuned RF")
    test_metrics = evaluate_regression(y_test_c, y_test_pred, "TEST - Tuned RF")

    return best_model, grid_search, train_metrics, val_metrics, test_metrics


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
    X_train_c, y_train_c = _prepare_model_pair(X_train, y_train)
    X_val_c, y_val_c = _prepare_model_pair(X_val, y_val)
    X_test_c, y_test_c = _prepare_model_pair(X_test, y_test)

    cv_splits = min(3, len(X_train_c) - 1)
    if cv_splits < 2:
        print("Not enough rows for cross-validation; fitting Gradient Boosting with default parameters instead.")
        best_model = gb_base.fit(X_train_c, y_train_c)
        y_train_pred = best_model.predict(X_train_c) if len(X_train_c) > 0 else np.array([])
        y_val_pred = best_model.predict(X_val_c) if len(X_val_c) > 0 else np.array([])
        y_test_pred = best_model.predict(X_test_c) if len(X_test_c) > 0 else np.array([])
        train_metrics = evaluate_regression(y_train_c, y_train_pred, "TRAINING - Tuned GB")
        val_metrics = evaluate_regression(y_val_c, y_val_pred, "VALIDATION - Tuned GB")
        test_metrics = evaluate_regression(y_test_c, y_test_pred, "TEST - Tuned GB")
        return best_model, None, train_metrics, val_metrics, test_metrics

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    grid_search = RandomizedSearchCV(
        gb_base, param_grid, n_iter=20, cv=tscv, n_jobs=-1,
        scoring='neg_mean_squared_error', verbose=1, random_state=42
    )

    print("Running hyperparameter tuning (20 iterations)...")
    grid_search.fit(X_train_c, y_train_c)

    best_model = cast(GradientBoostingRegressor, grid_search.best_estimator_)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

    # Predictions
    y_train_pred = best_model.predict(X_train_c)
    y_val_pred = best_model.predict(X_val_c)
    y_test_pred = best_model.predict(X_test_c)

    # Evaluate
    train_metrics = evaluate_regression(y_train_c, y_train_pred, "TRAINING - Tuned GB")
    val_metrics = evaluate_regression(y_val_c, y_val_pred, "VALIDATION - Tuned GB")
    test_metrics = evaluate_regression(y_test_c, y_test_pred, "TEST - Tuned GB")

    return best_model, grid_search, train_metrics, val_metrics, test_metrics


def train_with_mlflow_manual_logging(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    output_dir: str = 'Membangun_model/artifacts',
    metadata: Optional[Dict[str, object]] = None
):
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
    group_col = metadata.get('group_col') if metadata else 'Group'
    mlflow.set_experiment(f"Amazon_Demand_Forecasting_Tuning_{group_col}")
    
    print("\n" + "=" * 80)
    print("KRITERIA 2: MODEL BUILDING - SKILLED LEVEL (3 Poin)")
    print("MLflow Manual Logging with Hyperparameter Tuning")
    print("=" * 80 + "\n")
    
    results = {}
    baseline_results = {}

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
    if 'category_encoded' in X_train.columns:
        print(f"Detected category-encoded per-category setup with {metadata.get('n_groups') if metadata else 'unknown'} groups.")

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
        X_test_c = _clean_X_for_model(X_test)
        rf_model, rf_grid, rf_train_metrics, rf_val_metrics, rf_test_metrics = hyperparameter_tuning_random_forest(
            X_train, y_train, X_val, y_val, X_test, y_test, output_dir
        )

        rf_naive_val_metrics = evaluate_regression(y_val, _baseline_naive_prediction(X_val), "VALIDATION - Naive Baseline")
        rf_naive_test_metrics = evaluate_regression(y_test, _baseline_naive_prediction(X_test), "TEST - Naive Baseline")
        baseline_results['rf_naive'] = {
            'val_rmse': rf_naive_val_metrics['rmse'],
            'test_rmse': rf_naive_test_metrics['rmse']
        }

        params = {
            'model_type': 'RandomForestRegressor',
            'n_estimators': int(getattr(rf_model, 'n_estimators', 0)),
            'max_depth': str(getattr(rf_model, 'max_depth', None)),
            'group_col': metadata.get('group_col') if metadata else None,
            'group_value': metadata.get('group_value') if metadata else None,
            'target_name': metadata.get('target_name') if metadata else None,
            'n_groups': metadata.get('n_groups') if metadata else None
        }
        params = {k: (str(v) if v is not None else 'None') for k, v in params.items()}
        mlflow.log_params(params)

        try:
            if rf_grid is None:
                raise ValueError('Random Forest tuning did not run CV; skipping CV summary logging.')
            best_params = rf_grid.best_params_
            mlflow.log_params({'rf_best_' + k: str(v) for k, v in best_params.items()})
            cv_results = rf_grid.cv_results_
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

        mlflow.log_metrics({
            'train_mae': rf_train_metrics['mae'],
            'train_rmse': rf_train_metrics['rmse'],
            'train_r2': rf_train_metrics['r2'],
            'train_mape_pct': rf_train_metrics['mape_pct'],
            'val_mae': rf_val_metrics['mae'],
            'val_rmse': rf_val_metrics['rmse'],
            'val_r2': rf_val_metrics['r2'],
            'val_mape_pct': rf_val_metrics['mape_pct'],
            'test_mae': rf_test_metrics['mae'],
            'test_rmse': rf_test_metrics['rmse'],
            'test_r2': rf_test_metrics['r2'],
            'test_mape_pct': rf_test_metrics['mape_pct'],
            'baseline_val_rmse': rf_naive_val_metrics['rmse'],
            'baseline_test_rmse': rf_naive_test_metrics['rmse']
        })

        model_path = f'{output_dir}/rf_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        mlflow.log_artifact(model_path)

        try:
            feat_names = list(_clean_X_for_model(X_train).columns)
            fi_csv = _export_feature_importance(rf_model, feat_names, output_dir, 'rf')
            if fi_csv is not None:
                mlflow.log_artifact(str(fi_csv))

            importances = getattr(rf_model, 'feature_importances_', None)
            if importances is not None:
                importance_df = pd.DataFrame({
                    'feature': feat_names[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                fig, ax = plt.subplots(figsize=(8, max(3, len(importance_df) * 0.3)))
                sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
                ax.set_title('RF Feature Importances')
                fig.tight_layout()
                fi_path = Path(output_dir) / 'rf_feature_importances.png'
                fig.savefig(fi_path)
                plt.close(fig)
                mlflow.log_artifact(str(fi_path))
        except Exception:
            pass

        try:
            y_test_pred = rf_model.predict(X_test_c)
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
            'val_rmse': rf_val_metrics['rmse'],
            'test_rmse': rf_test_metrics['rmse'],
            'metrics': {
                'train': rf_train_metrics,
                'val': rf_val_metrics,
                'test': rf_test_metrics
            }
        }
    
    # ========== GRADIENT BOOSTING TUNING ==========
    with mlflow.start_run(run_name="gb_tuned"):
        X_test_c = _clean_X_for_model(X_test)
        gb_model, gb_grid, gb_train_metrics, gb_val_metrics, gb_test_metrics = hyperparameter_tuning_gradient_boosting(
            X_train, y_train, X_val, y_val, X_test, y_test, output_dir
        )

        gb_naive_val_metrics = evaluate_regression(y_val, _baseline_naive_prediction(X_val), "VALIDATION - Naive Baseline")
        gb_naive_test_metrics = evaluate_regression(y_test, _baseline_naive_prediction(X_test), "TEST - Naive Baseline")
        baseline_results['gb_naive'] = {
            'val_rmse': gb_naive_val_metrics['rmse'],
            'test_rmse': gb_naive_test_metrics['rmse']
        }

        params = {
            'model_type': 'GradientBoostingRegressor',
            'n_estimators': int(getattr(gb_model, 'n_estimators', 0)),
            'learning_rate': float(getattr(gb_model, 'learning_rate', 0.0)),
            'group_col': metadata.get('group_col') if metadata else None,
            'group_value': metadata.get('group_value') if metadata else None,
            'target_name': metadata.get('target_name') if metadata else None,
            'n_groups': metadata.get('n_groups') if metadata else None
        }
        params = {k: (str(v) if v is not None else 'None') for k, v in params.items()}
        mlflow.log_params(params)

        mlflow.log_metrics({
            'train_mae': gb_train_metrics['mae'],
            'train_rmse': gb_train_metrics['rmse'],
            'train_r2': gb_train_metrics['r2'],
            'train_mape_pct': gb_train_metrics['mape_pct'],
            'val_mae': gb_val_metrics['mae'],
            'val_rmse': gb_val_metrics['rmse'],
            'val_r2': gb_val_metrics['r2'],
            'val_mape_pct': gb_val_metrics['mape_pct'],
            'test_mae': gb_test_metrics['mae'],
            'test_rmse': gb_test_metrics['rmse'],
            'test_r2': gb_test_metrics['r2'],
            'test_mape_pct': gb_test_metrics['mape_pct'],
            'baseline_val_rmse': gb_naive_val_metrics['rmse'],
            'baseline_test_rmse': gb_naive_test_metrics['rmse']
        })

        try:
            if gb_grid is None:
                raise ValueError('Gradient Boosting tuning did not run CV; skipping CV summary logging.')
            best_params = gb_grid.best_params_
            mlflow.log_params({'gb_best_' + k: str(v) for k, v in best_params.items()})
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

        model_path = f'{output_dir}/gb_tuned_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(gb_model, f)
        mlflow.log_artifact(model_path)

        try:
            feat_names = list(_clean_X_for_model(X_train).columns)
            fi_csv = _export_feature_importance(gb_model, feat_names, output_dir, 'gb')
            if fi_csv is not None:
                mlflow.log_artifact(str(fi_csv))

            importances = getattr(gb_model, 'feature_importances_', None)
            if importances is not None:
                importance_df = pd.DataFrame({
                    'feature': feat_names[:len(importances)],
                    'importance': importances
                }).sort_values('importance', ascending=False)
                fig, ax = plt.subplots(figsize=(8, max(3, len(importance_df) * 0.3)))
                sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
                ax.set_title('GB Feature Importances')
                fig.tight_layout()
                fi_path = Path(output_dir) / 'gb_feature_importances.png'
                fig.savefig(fi_path)
                plt.close(fig)
                mlflow.log_artifact(str(fi_path))
        except Exception:
            pass

        try:
            y_test_pred = gb_model.predict(X_test_c)
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
            'val_rmse': gb_val_metrics['rmse'],
            'test_rmse': gb_test_metrics['rmse'],
            'metrics': {
                'train': gb_train_metrics,
                'val': gb_val_metrics,
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
        print(f"  Validation RMSE: {result['val_rmse']:.4f}")
        print(f"  Test RMSE: {result['test_rmse']:.4f}")

        if result['val_rmse'] < best_rmse:
            best_rmse = result['val_rmse']
            best_model = result['model']
            best_name = model_name

    print(f"\n{'=' * 80}")
    print(f"Best Tuned Model: {best_name} (Validation RMSE: {best_rmse:.4f})")
    print(f"{'=' * 80}\n")

    if baseline_results:
        naive_val_rmse = min(v['val_rmse'] for v in baseline_results.values())
        naive_test_rmse = min(v['test_rmse'] for v in baseline_results.values())
        print(f"Naive Baseline RMSE - Validation: {naive_val_rmse:.4f}, Test: {naive_test_rmse:.4f}")
        if best_rmse > naive_val_rmse:
            print("WARNING: Selected model is worse than the naive baseline on validation RMSE.")

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
    print("LOADING DEMAND DATA")
    print("=" * 80 + "\n")

    import os
    import sys

    group_col = os.environ.get('GROUP_COL', 'Category')
    group_value = os.environ.get('GROUP_VALUE')
    target_col = os.environ.get('TARGET_COL', 'Daily_Demand')
    min_group_size = int(os.environ.get('MIN_GROUP_SIZE', '30'))
    csv_path = os.environ.get(
        'DEMAND_CSV',
        'amazon_preprocessing/daily_demand_forecasting.csv'
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

    # Train with hyperparameter tuning
    best_model, results = train_with_mlflow_manual_logging(
        X_train, X_val, X_test, y_train, y_val, y_test, metadata=meta
    )
    
    print("Features:")
    print("  - Hyperparameter tuning dengan RandomizedSearchCV")
    print("  - Manual logging di MLflow (tidak autolog)")
    print("  - Metrik regresi: MAE, RMSE, R2, MAPE")
    print("  - Comparison antara RandomForestRegressor dan GradientBoostingRegressor")
    print("\nMLflow Tracking UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == '__main__':
    main()


def load_preprocessed_data(force_aggregation=None):
    csv_path = os.environ.get('DEMAND_CSV', 'amazon_preprocessing/daily_demand_forecasting.csv')
    group_col = os.environ.get('GROUP_COL', 'Category')
    group_value = os.environ.get('GROUP_VALUE')
    target_col = os.environ.get('TARGET_COL', 'Daily_Demand')
    min_group_size = int(os.environ.get('MIN_GROUP_SIZE', '30'))

    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_train_data(
        csv_path=csv_path,
        group_col=group_col,
        group_value=group_value,
        target_col=target_col,
        min_group_size=min_group_size
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
