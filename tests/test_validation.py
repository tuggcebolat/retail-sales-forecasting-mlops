
# Testlerde kullanacağımız kütüphaneler.
import pandas as pd
import pytest

# Test edeceğimiz fonksiyon.
from src.data.validation import validate_sales_data


# Geçerli verinin kabul edildiğini test ediyoruz.
def test_valid_data():

    # Aylık düzenli örnek veri oluşturuyoruz.
    df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01",
            "2026-03-01"
        ]),
        "sales": [100, 120, 140]
    })

    # Doğrulama fonksiyonunu çalıştırıyoruz.
    result = validate_sales_data(df)

    # Sonuçta üç kayıt bulunmasını bekliyoruz.
    assert len(result) == 3


# Eksik satış değerinin reddedildiğini test ediyoruz.
def test_missing_sales():

    # Eksik değer içeren veri oluşturuyoruz.
    df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-02-01"
        ]),
        "sales": [100, None]
    })

    # Fonksiyonun ValueError vermesini bekliyoruz.
    with pytest.raises(ValueError):
        validate_sales_data(df)


# Tekrarlanan tarihlerin reddedildiğini test ediyoruz.
def test_duplicate_dates():

    # Aynı tarih iki kez bulunuyor.
    df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-01-01"
        ]),
        "sales": [100, 120]
    })

    # Tekrarlanan tarih için hata bekliyoruz.
    with pytest.raises(ValueError):
        validate_sales_data(df)


# Eksik ayın tespit edildiğini test ediyoruz.
def test_missing_month():

    # Şubat ayı kasıtlı olarak eksik bırakılıyor.
    df = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01",
            "2026-03-01"
        ]),
        "sales": [100, 140]
    })

    # Tarih frekansı düzensiz olduğu için hata bekliyoruz.
    with pytest.raises(ValueError):
        validate_sales_data(df)
