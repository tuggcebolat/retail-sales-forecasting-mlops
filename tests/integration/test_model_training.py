import pandas as pd
import numpy as np
import pytest
from src.models.xgboost_model import recursive_forecast_xgboost
from src.models.model_training import train_selected_model


# Ortak test veri setimizi oluşturuyoruz.
@pytest.fixture
def sample_sales_data():

    # 60 aylık tarih indeksi oluşturuyoruz.
    dates = pd.date_range(
        start="2020-01-01",
        periods=60,
        freq="MS"
    )

    # Tekrarlanabilir rastgele veri üretiyoruz.
    rng = np.random.default_rng(42)

    # Örnek satış verisini oluşturuyoruz.
    df = pd.DataFrame({
        "sales": rng.integers(100, 500, size=60)
    }, index=dates)

    return df
def test_seasonal_naive(sample_sales_data):

    # Seasonal Naive modelini hazırlıyoruz.
    model = train_selected_model(
        sample_sales_data, "Seasonal Naive"
    )

    # Modelin 12 aylık veri içerdiğini doğruluyoruz.
    assert len(model) == 12

    # Son 12 aylık satış değerlerini doğruluyoruz.
    pd.testing.assert_series_equal(
        model,
        sample_sales_data["sales"].iloc[-12:]
    )

def test_sarima(sample_sales_data):

    # SARIMA modelini eğitiyoruz.
    model = train_selected_model(
        sample_sales_data, "SARIMA"
    )

    # Gelecek 3 ay için tahmin oluşturuyoruz.
    forecast = model.forecast(steps=3)

    # 3 adet tahmin üretildiğini doğruluyoruz.
    assert len(forecast) == 3

    # Tahminlerin sonlu sayılar olduğunu doğruluyoruz.
    assert np.isfinite(forecast).all()

def test_xgboost(sample_sales_data):

    # XGBoost modelini eğitiyoruz.
    model = train_selected_model(
        sample_sales_data, "XGBoost"
    )

    # Gelecek 3 ay için tahmin oluşturuyoruz.
    forecast = recursive_forecast_xgboost(
        model, sample_sales_data, horizon=3
    )

    # Tahmin sayısını doğruluyoruz.
    assert len(forecast) == 3

    # Tahminlerin sonlu sayılar olduğunu doğruluyoruz.
    assert np.isfinite(forecast).all()