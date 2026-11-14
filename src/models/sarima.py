
# Veri işlemleri için Pandas kütüphanesini içe aktarıyoruz.
import pandas as pd

# SARIMA modelini oluşturmak için SARIMAX sınıfını içe aktarıyoruz.
from statsmodels.tsa.statespace.sarimax import SARIMAX


# SARIMA modelini eğiten fonksiyonumuzu tanımlıyoruz.
def train_sarima_model(
    train_df: pd.DataFrame,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12)
):
    # Eğitim verisinde sales sütununun bulunmasını kontrol ediyoruz.
    if "sales" not in train_df.columns:
        raise ValueError("sales sütunu bulunamadı.")

    # Eğitim verisinin boş olup olmadığını kontrol ediyoruz.
    if train_df.empty:
        raise ValueError("Eğitim verisi boş olamaz.")

    # Orijinal DataFrame'i değiştirmeden satışları seçiyoruz.
    sales = train_df["sales"].copy()

    # SARIMA modelimizi önceki ayarlarımızla oluşturuyoruz.
    model = SARIMAX(
        sales,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Modelimizi eğitim verisi üzerinde eğitiyoruz.
    fitted_model = model.fit(disp=False)

    # Eğitilmiş model nesnesini geri döndürüyoruz.
    return fitted_model


# SARIMA ile gelecek dönem satışlarını tahmin ediyoruz.
def sarima_forecast(
    train_df: pd.DataFrame,
    horizon: int = 6,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12)
) -> pd.Series:

    # Tahmin ufkunun en az bir dönem olmasını sağlıyoruz.
    if horizon < 1:
        raise ValueError("Horizon en az 1 olmalıdır.")

    # Modeli ortak eğitim fonksiyonumuzla eğitiyoruz.
    fitted_model = train_sarima_model(
        train_df=train_df,
        order=order,
        seasonal_order=seasonal_order
    )

    # Eğitilmiş modelle gelecek dönem tahminlerini üretiyoruz.
    forecast_result = fitted_model.get_forecast(steps=horizon)

    # Tahmin edilen değerleri alıyoruz.
    predictions = forecast_result.predicted_mean

    # Tahmin serisine açıklayıcı bir isim veriyoruz.
    predictions.name = "predicted_sales"

    # Tahminleri Pandas Series olarak döndürüyoruz.
    return predictions
