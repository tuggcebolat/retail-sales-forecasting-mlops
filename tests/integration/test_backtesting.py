
# Test verisi oluşturmak için Pandas.
import pandas as pd

# Test edeceğimiz backtesting fonksiyonu.
from src.models.backtesting import rolling_backtest


# Backtesting dönemlerinin doğru oluşturulduğunu test ediyoruz.
def test_rolling_backtest_creates_expected_windows():

    # 36 aylık düzenli örnek veri oluşturuyoruz.
    dates = pd.date_range(
        start="2023-01-01",
        periods=36,
        freq="MS"
    )

    # Satış değerlerini örnek olarak oluşturuyoruz.
    df = pd.DataFrame(
        {"sales": list(range(100, 136))},
        index=dates
    )

    # 24 aylık eğitim ve 6 aylık test pencereleri kullanıyoruz.
    results = rolling_backtest(
        df,
        initial_train_size=24,
        horizon=6,
        step=6
    )

    # Toplam iki değerlendirme dönemi oluşmasını bekliyoruz.
    assert len(results) == 2

    # İlk test döneminin Ocak 2025'te başlamasını bekliyoruz.
    assert results.iloc[0]["test_start"] == pd.Timestamp("2025-01-01")

    # İkinci test döneminin Temmuz 2025'te başlamasını bekliyoruz.
    assert results.iloc[1]["test_start"] == pd.Timestamp("2025-07-01")
