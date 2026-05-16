"""
Train models with manual MLflow logging suitable for remote tracking (e.g., DagsHub).

Usage:
  python Membangun_model/train_with_mlflow_dagshub.py --experiment "MyExperiment" --tracking_uri $MLFLOW_TRACKING_URI

Set `MLFLOW_TRACKING_URI` environment variable to your DagsHub MLflow URI (e.g. https://dagshub.com/<owner>/<repo>.mlflow)
and ensure any auth tokens are set per your DagsHub instructions.

This script performs manual logging (no autolog), and logs at least two additional artifacts:
- feature importances plot
- residuals scatter plot
"""

import os
import argparse
from pathlib import Path
import pickle

# Ensure project root is on sys.path so local package imports work when running script directly
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import mlflow

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import helper functions from modelling.py
from Membangun_model.modelling import load_preprocessed_data, evaluate_regression


def plot_feature_importances(importances, feature_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.3)))
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title('Feature Importances')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_residuals(y_true, y_pred, out_path: Path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual (True - Pred)')
    ax.set_title('Residuals vs Predicted')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def train_and_log(tracking_uri: str, experiment_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()

    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            # Log params manually
            mlflow.log_params({'model_type': name, 'n_estimators': getattr(model, 'n_estimators', None)})

            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # Evaluate and log metrics (manual)
            train_metrics = evaluate_regression(y_train, y_train_pred, dataset_name='TRAIN')
            val_metrics = evaluate_regression(y_val, y_val_pred, dataset_name='VAL')
            test_metrics = evaluate_regression(y_test, y_test_pred, dataset_name='TEST')

            # Additional metrics not always provided by autolog
            # e.g., MAPE (percent) already returned by evaluate_regression as 'mape_pct'
            extra_metrics = {
                'train_mape_pct': train_metrics.get('mape_pct'),
                'val_mape_pct': val_metrics.get('mape_pct'),
                'test_mape_pct': test_metrics.get('mape_pct')
            }

            all_metrics = {
                'train_mae': train_metrics['mae'],
                'train_rmse': train_metrics['rmse'],
                'train_r2': train_metrics['r2'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2']
            }
            all_metrics.update(extra_metrics)

            mlflow.log_metrics(all_metrics)

            # Save model artifact
            model_path = Path(output_dir) / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path))

            # Feature importances (if available)
            try:
                importances = getattr(model, 'feature_importances_', None)
                if importances is not None:
                    feat_names = list(X_train.columns)
                    fi_path = Path(output_dir) / f"{name}_feature_importances.png"
                    plot_feature_importances(importances, feat_names, fi_path)
                    mlflow.log_artifact(str(fi_path))
            except Exception as e:
                print(f"Could not log feature importances: {e}")

            # Residuals plot
            try:
                resid_path = Path(output_dir) / f"{name}_residuals.png"
                plot_residuals(y_test, y_test_pred, resid_path)
                mlflow.log_artifact(str(resid_path))
            except Exception as e:
                print(f"Could not create residuals plot: {e}")

            # Also log a small CSV with sample predictions
            try:
                sample_df = X_test.head(100).copy()
                sample_df['y_true'] = y_test.reset_index(drop=True).head(100)
                sample_df['y_pred'] = y_test_pred[:100]
                sample_pred_path = Path(output_dir) / f"{name}_predictions_sample.csv"
                sample_df.to_csv(sample_pred_path, index=False)
                mlflow.log_artifact(str(sample_pred_path))
            except Exception as e:
                print(f"Could not log sample predictions: {e}")

            print(f"Run {name} logged to MLflow at {mlflow.get_tracking_uri()}")

            # Track best model based on validation RMSE
            val_rmse = val_metrics.get('rmse', None)
            if val_rmse is not None:
                try:
                    best_info = getattr(train_and_log, 'best_info', None)
                    if best_info is None or val_rmse < best_info['val_rmse']:
                        # save best model
                        best_model_path = Path(output_dir) / f"best_model_{name}.pkl"
                        with open(best_model_path, 'wb') as bf:
                            pickle.dump(model, bf)
                        mlflow.log_artifact(str(best_model_path), artifact_path='best_model')
                        train_and_log.best_info = {'val_rmse': val_rmse, 'model_name': name, 'path': str(best_model_path)}
                        print(f"New best model: {name} (val_rmse={val_rmse:.4f}) saved to {best_model_path}")
                except Exception as e:
                    print(f"Could not save best model: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_uri', default=os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db'))
    parser.add_argument('--experiment', default='Amazon_Daily_Revenue_Forecasting_Advanced')
    parser.add_argument('--output', default='Membangun_model/artifacts')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Using MLflow tracking URI: {args.tracking_uri}")
    train_and_log(args.tracking_uri, args.experiment, args.output)
