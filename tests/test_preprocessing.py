
# Test verisi oluşturmak için Pandas'ı içe aktarıyoruz.
import pandas as pd

# Test edeceğimiz preprocessing fonksiyonunu içe aktarıyoruz.
from src.data.preprocessing import preprocess_sales_data


# Tarih sütununun DatetimeIndex'e dönüştürüldüğünü test ediyoruz.
def test_preprocessing_datetime_index():

    # ARRANGE: Örnek aylık satış verisini oluşturuyoruz.
    df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01",
            "2026-03-01"
        ]),
        "sales": [100, 120, 140]
    })

    # ACT: Ön işleme fonksiyonunu çalıştırıyoruz.
    result = preprocess_sales_data(df)

    # ASSERT: Index'in DatetimeIndex olduğunu doğruluyoruz.
    assert isinstance(result.index, pd.DatetimeIndex)
# Orijinal DataFrame'in değişmediğini test ediyoruz.
def test_preprocessing_does_not_modify_original():

    # ARRANGE: Test verisini hazırlıyoruz.
    df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01"
        ]),
        "sales": [100, 200]
    })

    # ACT: Preprocessing fonksiyonunu çalıştırıyoruz.
    result = preprocess_sales_data(df)

    # ASSERT: Orijinal veride tarih sütunu hâlâ bulunmalı.
    assert "observation_date" in df.columns

    # ASSERT: İşlenmiş veride tarih artık index olmalı.
    assert "observation_date" not in result.columns

    # ASSERT: Orijinal veri sayısal index'ini korumalı.
    assert isinstance(df.index, pd.RangeIndex)
