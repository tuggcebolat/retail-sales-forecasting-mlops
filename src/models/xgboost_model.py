
# Veri işleme işlemleri için pandas kütüphanesini içe aktarıyoruz.
import pandas as pd

# XGBoost regresyon algoritmasını içe aktarıyoruz.
from xgboost import XGBRegressor


# XGBoost modelini eğitecek fonksiyonumuzu tanımlıyoruz.
def train_xgboost_model(X_train, y_train):

    # Başlangıç parametreleriyle modelimizi oluşturuyoruz.
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # Modelin geçmiş satış özelliklerinden
    # gerçek satış değerlerini öğrenmesini sağlıyoruz.
    model.fit(X_train, y_train)

    # Eğitilmiş modeli döndürüyoruz.
    return model


# Eğitilmiş XGBoost modelini kullanarak gelecek ayların
# satışlarını tahmin edecek fonksiyonumuzu tanımlıyoruz.
def recursive_forecast_xgboost(model, train_df, horizon=6):

    # Orijinal eğitim verisini değiştirmeden
    # geçmiş satışları bir listeye dönüştürüyoruz.
    history = train_df["sales"].tolist()

    # Üreteceğimiz tahminleri saklamak için boş liste oluşturuyoruz.
    predictions = []

    # Tahmin yapacağımız tarihleri saklamak için liste oluşturuyoruz.
    forecast_dates = []

    # Eğitim verisindeki son tarihi alıyoruz.
    last_date = train_df.index[-1]

    # İlk tahmin edeceğimiz ayı belirliyoruz.
    next_date = last_date + pd.DateOffset(months=1)

    # Tahmin etmek istediğimiz ay sayısı kadar döngü oluşturuyoruz.
    for step in range(horizon):

        # Geçmiş satışlardan gecikmeli özellikleri hesaplıyoruz.
        lag_1 = history[-1]
        lag_3 = history[-3]
        lag_12 = history[-12]

        # Son 3 ayın satış ortalamasını hesaplıyoruz.
        rolling_mean_3 = sum(history[-3:]) / 3

        # Son 6 ayın satış ortalamasını hesaplıyoruz.
        rolling_mean_6 = sum(history[-6:]) / 6

        # Tahmin edeceğimiz ayın takvim özelliklerini oluşturuyoruz.
        month = next_date.month
        quarter = next_date.quarter
        year = next_date.year

        # Modelimizin eğitimde kullandığı 8 özelliği oluşturuyoruz.
        feature_values = {
            "lag_1": lag_1,
            "lag_3": lag_3,
            "lag_12": lag_12,
            "rolling_mean_3": rolling_mean_3,
            "rolling_mean_6": rolling_mean_6,
            "month": month,
            "quarter": quarter,
            "year": year
        }

        # Özellikleri tek satırlık DataFrame'e dönüştürüyoruz.
        X_next = pd.DataFrame([feature_values])

        # Özelliklerin sırasını eğitim verisiyle aynı tutuyoruz.
        X_next = X_next[model.feature_names_in_]

        # XGBoost modelimizle satış tahmini yapıyoruz.
        prediction = model.predict(X_next)

        # Tahmin dizisindeki ilk değeri alıyoruz.
        predicted_sales = float(prediction[0])

        # Tahmini sonraki ayların özelliklerinde kullanmak için
        # geçmiş satış listesine ekliyoruz.
        history.append(predicted_sales)

        # Ürettiğimiz tahmini sonuç listesine ekliyoruz.
        predictions.append(predicted_sales)

        # Tahmin yaptığımız ayın tarihini kaydediyoruz.
        forecast_dates.append(next_date)

        # Bir sonraki tahmin ayına geçiyoruz.
        next_date = next_date + pd.DateOffset(months=1)

    # Tahmin sonuçlarını tarih indeksli pandas Series'e dönüştürüyoruz.
    forecast = pd.Series(
        predictions,
        index=pd.DatetimeIndex(forecast_dates),
        name="xgboost_forecast"
    )

    # Tahmin sonuçlarını döndürüyoruz.
    return forecast
