from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.features.feature_engineering import create_features
from src.models.xgboost_model import train_xgboost_model


def train_sarima(df):

    # Initialize the SARIMA model.
    model = SARIMAX(
        df["sales"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    # Train the model using all historical data.
    fitted_model = model.fit(disp=False)

    # Return the trained model.
    return fitted_model


def train_xgboost(df):

    # Generate features using historical data.
    featured_df = create_features(df)

    # Prepare independent variables.
    X = featured_df.drop(columns=["sales"])

    # Prepare the target variable.
    y = featured_df["sales"]

    # Train the final XGBoost model.
    model = train_xgboost_model(X, y)

    return model

def train_seasonal_naive(df):

    # Store the last 12 months of sales.
    model = df["sales"].iloc[-12:].copy()

    # Return the historical seasonal values.
    return model
def train_selected_model(df, model_name):

    # Train the selected model.
    if model_name == "SARIMA":
        return train_sarima(df)

    elif model_name == "XGBoost":
        return train_xgboost(df)

    elif model_name == "Seasonal Naive":
        return train_seasonal_naive(df)

    # Raise an error for unsupported models.
    else:
        raise ValueError(f"Unknown model: {model_name}")