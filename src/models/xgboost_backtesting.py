
# XGBoost için eğitim özelliklerini oluşturuyoruz.
from src.features.feature_engineering import create_features

# Model eğitimi ve çok adımlı tahmin fonksiyonlarını çağırıyoruz.
from src.models.xgboost_model import (
    train_xgboost_model,
    recursive_forecast_xgboost
)

# Tahminlerin hata metriklerini hesaplıyoruz.
from src.models.evaluation import evaluate_forecast

# Backtesting sonuçlarını tabloya dönüştürmek için kullanıyoruz.
import pandas as pd


def xgboost_rolling_backtest(
    df,
    initial_train_size=312,
    horizon=6,
    step=6
):
    # Her backtesting döneminin sonuçlarını saklayacağız.
    results = []

    # İlk eğitim penceremizin uzunluğunu belirliyoruz.
    train_end = initial_train_size

    # Yeterli eğitim ve test verisi bulunduğu sürece devam ediyoruz.
    while train_end + horizon <= len(df):

        # Sadece geçmiş verileri eğitim için ayırıyoruz.
        train_df = df.iloc[:train_end]

        # Sonraki 6 aylık gerçek satışları test için ayırıyoruz.
        test_df = df.iloc[train_end:train_end + horizon]

        # Eğitim verisinden lag, rolling ve takvim özelliklerini üretiyoruz.
        feature_train_df = create_features(train_df)

        # Modelin bağımsız değişkenlerini hazırlıyoruz.
        X_train = feature_train_df.drop(columns=["sales"])

        # Modelin tahmin edeceği gerçek satış değerlerini ayırıyoruz.
        y_train = feature_train_df["sales"]

        # Her backtesting turunda modeli yeniden eğitiyoruz.
        model = train_xgboost_model(X_train, y_train)

        # Sonraki 6 ayın satış tahminlerini üretiyoruz.
        predictions = recursive_forecast_xgboost(
            model=model,
            train_df=train_df,
            horizon=horizon
        )

        # Gerçek satışlarla tahminleri karşılaştırıyoruz.
        metrics = evaluate_forecast(
            test_df["sales"],
            predictions
        )

        # Dönemin tarihlerini ve hata metriklerini kaydediyoruz.
        results.append({
            "model": "XGBoost",
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "sMAPE": metrics["sMAPE"]
        })

        # Eğitim penceresini 6 ay ileri taşıyoruz.
        train_end += step

    # Bütün dönemlerin sonuçlarını DataFrame olarak döndürüyoruz.
    return pd.DataFrame(results)
