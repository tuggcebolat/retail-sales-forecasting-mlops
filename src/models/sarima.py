
# 1. Veri işlemleri için Pandas kütüphanesini içe aktarıyoruz.
import pandas as pd

# 2. SARIMA modelini oluşturmak için SARIMAX sınıfını çağırıyoruz.
from statsmodels.tsa.statespace.sarimax import SARIMAX


# 3. SARIMA modeliyle gelecek dönem satışlarını tahmin ediyoruz.
def sarima_forecast(
    train_df: pd.DataFrame,
    horizon: int = 6,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12)
) -> pd.Series:

    # 4. Tahmin edilecek dönem sayısını kontrol ediyoruz.
    if horizon < 1:
        raise ValueError("Horizon en az 1 olmalıdır.")

    # 5. Eğitim verisinde sales sütunu bulunmalı.
    if "sales" not in train_df.columns:
        raise ValueError("sales sütunu bulunamadı.")

    # 6. Eğitim verisinin boş olmaması gerekiyor.
    if train_df.empty:
        raise ValueError("Eğitim verisi boş olamaz.")

    # 7. Orijinal veriyi değiştirmeden satış sütununu seçiyoruz.
    sales = train_df["sales"].copy()

    # 8. SARIMA modelini tanımlıyoruz.
    model = SARIMAX(
        sales,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # 9. Modeli geçmiş satış verileriyle eğitiyoruz.
    fitted_model = model.fit(disp=False)

    # 10. Gelecek horizon dönem için tahmin sonuç nesnesi oluşturuyoruz.
    forecast_result = fitted_model.get_forecast(
        steps=horizon
    )

    # 11. Tahmin sonuç nesnesinden sayısal tahminleri alıyoruz.
    predictions = forecast_result.predicted_mean

    # 12. Tahmin serisine açıklayıcı bir isim veriyoruz.
    predictions.name = "predicted_sales"

    # 13. Sayısal tahminleri pandas Series olarak döndürüyoruz.
    return predictions
