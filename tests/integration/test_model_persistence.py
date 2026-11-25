import pandas as pd
import numpy as np

from src.models.model_training import train_selected_model
from src.models.model_persistence import save_selected_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def test_model_persistence(tmp_path):

    # 60 aylık örnek satış verisi oluşturuyoruz.
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")

    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {"sales": rng.integers(100, 500, size=60)},
        index=dates
    )

    # Modelimizi eğitiyoruz.
    model = train_selected_model(df, "SARIMA")

    # Kaydetmeden önce tahmin üretiyoruz.
    original_forecast = model.forecast(steps=6)

    # Geçici dosya yolunu oluşturuyoruz.
    file_path = tmp_path / "sarima.pkl"

    # Modelimizi kaydediyoruz.
    save_selected_model(model, "SARIMA", file_path)

    # Dosyanın oluşturulduğunu doğruluyoruz.
    assert file_path.exists()

    # Kaydedilen modeli yeniden yüklüyoruz.
    loaded_model = SARIMAXResults.load(file_path)

    # Yeniden tahmin üretiyoruz.
    loaded_forecast = loaded_model.forecast(steps=6)

    # İki tahmin sonucunun aynı olduğunu doğruluyoruz.
    pd.testing.assert_series_equal(
        original_forecast,
        loaded_forecast
    )