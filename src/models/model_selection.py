import pandas as pd


def summarize_model_performance(df):

    # Group results by model and calculate average errors.
    summary = df.groupby("model")[["MAE", "RMSE", "sMAPE"]].mean()

    # Sort models by average sMAPE.
    summary = summary.sort_values("sMAPE")

    # Return the performance summary.
    return summary.reset_index()

def select_best_model(summary):

    # Find the model with the lowest average sMAPE.
    best_model = summary.loc[summary["sMAPE"].idxmin()]

    # Return the best model's name.
    return best_model["model"]

def validate_test_periods(df):

    # Define the columns representing each test period.
    columns = ["test_start", "test_end"]

    # Get the unique test periods for each model.
    periods = [
        set(group[columns].itertuples(index=False, name=None))
        for _, group in df.groupby("model")
    ]

    # Check whether all models share the same test periods.
    if not periods or any(p != periods[0] for p in periods):
        raise ValueError("Models have different test periods.")
import numpy as np

def validate_metrics(df):

    # Select the evaluation metrics.
    metrics = df[["MAE", "RMSE", "sMAPE"]]

    # Check for missing or infinite values.
    if not np.isfinite(metrics.to_numpy()).all():
        raise ValueError("Invalid metric values.")

    # Check for negative error values.
    if (metrics < 0).any().any():
        raise ValueError("Negative metric values.")
def compare_and_select_models(df):

    # Validate the test periods.
    validate_test_periods(df)

    # Validate the evaluation metrics.
    validate_metrics(df)

    # Calculate average model performance.
    summary = summarize_model_performance(df)

    # Select the best-performing model.
    best_model = select_best_model(summary)

    # Return the summary and the selected model.
    return summary, best_model