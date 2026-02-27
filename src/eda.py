from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, TARGET_COL

sns.set_theme(style="whitegrid")


def summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    if "airline" in df.columns and TARGET_COL in df.columns:
        tables["fare_by_airline"] = df.groupby("airline", as_index=False)[TARGET_COL].agg(["mean", "median", "count"])

    if {"source", "destination", TARGET_COL}.issubset(df.columns):
        tables["fare_by_route"] = df.groupby(["source", "destination"], as_index=False)[TARGET_COL].mean()

    if "season" in df.columns and TARGET_COL in df.columns:
        tables["fare_by_season"] = df.groupby("season", as_index=False)[TARGET_COL].mean()

    return tables


def kpi_airline_average_fare(df: pd.DataFrame) -> pd.DataFrame:
    if not {"airline", TARGET_COL}.issubset(df.columns):
        return pd.DataFrame()
    return df.groupby("airline", as_index=False)[TARGET_COL].mean().sort_values(TARGET_COL, ascending=False)


def kpi_popular_routes(df: pd.DataFrame) -> pd.DataFrame:
    if not {"source", "destination"}.issubset(df.columns):
        return pd.DataFrame()
    routes = df.groupby(["source", "destination"], as_index=False).size()
    return routes.rename(columns={"size": "flight_count"}).sort_values("flight_count", ascending=False)


def kpi_top_expensive_routes(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if not {"source", "destination", TARGET_COL}.issubset(df.columns):
        return pd.DataFrame()
    return (
        df.groupby(["source", "destination"], as_index=False)[TARGET_COL]
        .mean()
        .sort_values(TARGET_COL, ascending=False)
        .head(top_n)
    )


def _save(fig_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / fig_name
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_distributions(df: pd.DataFrame) -> Dict[str, Path]:
    saved = {}
    for col, file_name in [(TARGET_COL, "total_fare_distribution.png"), ("base_fare", "base_fare_distribution.png"), ("tax_surcharge", "tax_surcharge_distribution.png")]:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution: {col}")
            saved[col] = _save(file_name)
    return saved


def plot_airline_boxplot(df: pd.DataFrame) -> Path | None:
    if not {"airline", TARGET_COL}.issubset(df.columns):
        return None
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="airline", y=TARGET_COL)
    plt.xticks(rotation=45, ha="right")
    plt.title("Fare Variation by Airline")
    return _save("fare_variation_by_airline.png")


def plot_monthly_or_seasonal_trends(df: pd.DataFrame) -> Dict[str, Path]:
    saved = {}

    if {"month", TARGET_COL}.issubset(df.columns):
        monthly = df.groupby("month", as_index=False)[TARGET_COL].mean().sort_values("month")
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=monthly, x="month", y=TARGET_COL, marker="o")
        plt.title("Average Fare by Month")
        saved["month"] = _save("average_fare_by_month.png")

    if {"season", TARGET_COL}.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="season", y=TARGET_COL)
        plt.title("Fare Variation Across Seasons")
        saved["season"] = _save("fare_variation_by_season.png")

    return saved


def plot_corr_heatmap(df: pd.DataFrame) -> Path | None:
    numeric_df = df.select_dtypes(include=["number"]) 
    if numeric_df.shape[1] < 2:
        return None

    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    return _save("feature_correlation_heatmap.png")


def run_eda(df: pd.DataFrame) -> Dict[str, object]:
    outputs: Dict[str, object] = {
        "summary_tables": summary_tables(df),
        "kpi_airline": kpi_airline_average_fare(df),
        "kpi_popular_routes": kpi_popular_routes(df),
        "kpi_top_expensive_routes": kpi_top_expensive_routes(df),
    }

    outputs["distribution_plots"] = plot_distributions(df)
    outputs["airline_boxplot"] = plot_airline_boxplot(df)
    outputs["trend_plots"] = plot_monthly_or_seasonal_trends(df)
    outputs["corr_heatmap"] = plot_corr_heatmap(df)
    return outputs
