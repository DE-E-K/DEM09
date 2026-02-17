from __future__ import annotations

from main import run_pipeline


def task_preprocess(data_path: str | None = None):
    from src.data_preprocessing import load_and_preprocess

    return load_and_preprocess(path=data_path, save_processed=True)


def task_train_and_evaluate(data_path: str | None = None):
    return run_pipeline(data_path=data_path)


def task_retrain(data_path: str | None = None):
    return run_pipeline(data_path=data_path)
