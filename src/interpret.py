from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR


def _feature_names_from_pipeline(model_pipeline: Pipeline):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, file_name: str = "predicted_vs_actual.png"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("Actual Fare")
    plt.ylabel("Predicted Fare")
    plt.title("Predicted vs Actual Fares")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=150)
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, file_name: str = "residual_plot.png"):
    residuals = y_true - y_pred
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Residual Diagnostics")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=150)
    plt.close()


def extract_linear_coefficients(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = _feature_names_from_pipeline(model_pipeline)

    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    coeffs = np.ravel(model.coef_)
    df = pd.DataFrame({"feature": feature_names, "coefficient": coeffs})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False)


def extract_tree_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    model = model_pipeline.named_steps["model"]
    feature_names = _feature_names_from_pipeline(model_pipeline)

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return df.sort_values("importance", ascending=False)


def plot_feature_importance(importance_df: pd.DataFrame, file_name: str = "feature_importance.png", top_n: int = 20):
    if importance_df.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    value_col = "importance" if "importance" in top_df.columns else "abs_coefficient"
    plt.barh(top_df["feature"], top_df[value_col])
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=150)
    plt.close()


def interpret_model(model_name: str, model_pipeline: Pipeline) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}

    if model_name in {"LinearRegression", "Ridge", "Lasso"}:
        coef_df = extract_linear_coefficients(model_pipeline)
        plot_feature_importance(coef_df, file_name=f"{model_name.lower()}_coefficients.png")
        outputs["coefficients"] = coef_df
    else:
        imp_df = extract_tree_feature_importance(model_pipeline)
        plot_feature_importance(imp_df, file_name=f"{model_name.lower()}_feature_importance.png")
        outputs["feature_importance"] = imp_df

    return outputs
