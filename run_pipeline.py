
# ============================================================
# RETAIL SALES FORECASTING - MLOPS PIPELINE
# ============================================================

# 1. Veri analizi ve DataFrame işlemleri için Pandas'ı içe aktarıyoruz.
import pandas as pd

# 2. FRED üzerinden satış verilerini çeken fonksiyonu çağırıyoruz.
from src.data.ingestion import load_sales_data

# 3. Veri kalitesini kontrol eden fonksiyonu çağırıyoruz.
from src.data.validation import validate_sales_data

# 4. Veriyi zaman serisi formatına hazırlayan fonksiyonu çağırıyoruz.
from src.data.preprocessing import preprocess_sales_data

# 5. EDA analiz ve görselleştirme fonksiyonlarını içe aktarıyoruz.
from src.data.eda import (
    analyze_sales_data,
    plot_sales_trend,
    plot_rolling_mean,
    analyze_monthly_seasonality,
    plot_seasonal_decomposition,
)

# 6. Seasonal Naive tahmin modelimizi içe aktarıyoruz.
from src.models.baseline import seasonal_naive_forecast

# 7. Yeni SARIMA tahmin modelimizi içe aktarıyoruz.
from src.models.sarima import sarima_forecast

# 8. MAE, RMSE ve sMAPE metriklerini hesaplayan fonksiyonu çağırıyoruz.
from src.models.evaluation import evaluate_forecast

# 9. Rolling-Origin Backtesting fonksiyonumuzu içe aktarıyoruz.
from src.models.backtesting import rolling_backtest


# 10. Projemizin bütün aşamalarını yöneten ana fonksiyonu tanımlıyoruz.
def main():

    # ========================================================
    # AŞAMA 1 - DATA INGESTION
    # ========================================================

    # 11. FRED veri kaynağından aylık satış verilerini çekiyoruz.
    df = load_sales_data()

    # ========================================================
    # AŞAMA 2 - DATA VALIDATION
    # ========================================================

    # 12. Eksik değer, negatif satış, tekrar eden tarih gibi
    # veri kalitesi kurallarını kontrol ediyoruz.
    validated_df = validate_sales_data(df)

    # ========================================================
    # AŞAMA 3 - DATA PREPROCESSING
    # ========================================================

    # 13. Tarih sütununu DatetimeIndex yapısına dönüştürüyor,
    # verileri sıralıyor ve aylık frekansı tanımlıyoruz.
    processed_df = preprocess_sales_data(validated_df)

    # 14. İlk üç aşamanın tamamlandığını terminale bildiriyoruz.
    print("Veri çekme, doğrulama ve ön işleme işlemleri tamamlandı.")

    # 15. İşlenmiş verinin ilk 5 satırını gösteriyoruz.
    print(processed_df.head())

    # 16. Index'in Python sınıfını kontrol ediyoruz.
    print(f"Index sınıfı: {type(processed_df.index)}")

    # 17. Index'in veri tipini kontrol ediyoruz.
    print(f"Index türü: {processed_df.index.dtype}")

    # 18. Verinin aylık frekansa sahip olduğunu doğruluyoruz.
    print(f"Veri frekansı: {processed_df.index.freq}")

    # ========================================================
    # AŞAMA 4 - EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================================

    # 19. Satış verisinin özet istatistiklerini hesaplıyoruz.
    statistics = analyze_sales_data(processed_df)

    # 20. Ortalama, standart sapma, minimum ve maksimum gibi
    # istatistikleri terminalde gösteriyoruz.
    print("\nSatış Verisi İstatistikleri:")
    print(statistics)

    # 21. Satışların zaman içindeki değişimini görselleştiriyoruz.
    plot_sales_trend(processed_df)

    # 22. 12 aylık hareketli ortalamayı görselleştirerek
    # uzun dönemli satış trendini inceliyoruz.
    plot_rolling_mean(processed_df)

    # 23. Her ayın mevsimsel satış profilini hesaplıyoruz.
    monthly_profile = analyze_monthly_seasonality(processed_df)

    # 24. Aylık mevsimsellik oranlarını terminalde gösteriyoruz.
    print("\nAylık mevsimsellik profili:")
    print(monthly_profile.round(3))

    # 25. Satış serisini trend, mevsimsellik ve kalıntı
    # bileşenlerine ayırarak görselleştiriyoruz.
    plot_seasonal_decomposition(processed_df)

    # ========================================================
    # AŞAMA 5 - TRAIN / TEST SPLIT
    # ========================================================

    # 26. Son 12 aylık gerçek satışları test setine ayırıyoruz.
    test_df = processed_df.iloc[-12:]

    # 27. Son 12 ay dışındaki geçmiş verileri eğitim seti
    # olarak kullanıyoruz.
    train_df = processed_df.iloc[:-12]

    # 28. Eğitim ve test dönemlerini terminalde gösteriyoruz.
    print("\nEğitim ve Test Dönemleri:")
    print(f"Eğitim başlangıcı: {train_df.index.min()}")
    print(f"Eğitim bitişi: {train_df.index.max()}")
    print(f"Test başlangıcı: {test_df.index.min()}")
    print(f"Test bitişi: {test_df.index.max()}")

    # ========================================================
    # AŞAMA 6 - SEASONAL NAIVE MODEL
    # ========================================================

    # 29. Geçen yılın aynı ayındaki satışları kullanarak
    # test dönemindeki 12 ay için tahmin üretiyoruz.
    naive_predictions = seasonal_naive_forecast(
        train_df,
        horizon=12
    )

    # 30. Seasonal Naive tahminlerini gerçek satışlarla
    # karşılaştırarak hata metriklerini hesaplıyoruz.
    naive_metrics = evaluate_forecast(
        test_df["sales"],
        naive_predictions
    )

    # 31. Seasonal Naive modelinin metriklerini gösteriyoruz.
    print("\nSeasonal Naive Model Performansı:")

    for metric, value in naive_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ========================================================
    # AŞAMA 7 - SARIMA MODEL
    # ========================================================

    # 32. SARIMA modelini yalnızca eğitim verisiyle eğitiyoruz.
    # order=(p,d,q) normal ARIMA parametrelerini belirtir.
    # seasonal_order=(P,D,Q,s) mevsimsel parametreleri belirtir.
    sarima_predictions = sarima_forecast(
        train_df=train_df,
        horizon=12,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    # 33. SARIMA tahminlerini gerçek satışlarla karşılaştırarak
    # MAE, RMSE ve sMAPE metriklerini hesaplıyoruz.
    sarima_metrics = evaluate_forecast(
        test_df["sales"],
        sarima_predictions
    )

    # 34. SARIMA modelinin performans metriklerini gösteriyoruz.
    print("\nSARIMA Model Performansı:")

    for metric, value in sarima_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ========================================================
    # AŞAMA 8 - MODEL COMPARISON
    # ========================================================

    # 35. Gerçek satışları ve iki modelin tahminlerini
    # aynı DataFrame içerisinde birleştiriyoruz.
    comparison = pd.DataFrame({
        "Actual Sales": test_df["sales"],
        "Seasonal Naive": naive_predictions,
        "SARIMA": sarima_predictions
    })

    # 36. Aylık tahmin karşılaştırmasını terminalde gösteriyoruz.
    print("\nGerçek ve Tahmin Edilen Satışlar:")
    print(comparison)

    # 37. İki modelin hata metriklerini karşılaştırmak için
    # yeni bir DataFrame oluşturuyoruz.
    metrics_comparison = pd.DataFrame({
        "Seasonal Naive": naive_metrics,
        "SARIMA": sarima_metrics
    })

    # 38. Modellerin performanslarını yan yana gösteriyoruz.
    print("\nModel Performans Karşılaştırması:")
    print(metrics_comparison.round(4))

    # ========================================================
    # AŞAMA 9 - ROLLING-ORIGIN BACKTESTING
    # ========================================================

    # 39. Seasonal Naive modelini farklı geçmiş dönemlerde
    # 6 aylık tahmin pencereleriyle değerlendiriyoruz.
    backtest_results = rolling_backtest(
        processed_df,
        initial_train_size=312,
        horizon=6,
        step=6
    )

    # 40. Her test döneminin MAE, RMSE ve sMAPE
    # sonuçlarını terminalde gösteriyoruz.
    print("\nRolling Backtesting Sonuçları:")
    print(backtest_results.to_string(index=False))

    # 41. Tüm backtesting dönemlerinin ortalama
    # performans metriklerini hesaplayıp gösteriyoruz.
    print("\nOrtalama Backtesting Performansı:")

    print(
        backtest_results[["MAE", "RMSE", "sMAPE"]]
        .mean()
        .round(2)
    )


# 42. Bu dosya doğrudan çalıştırıldığında main() fonksiyonunu
# başlatarak bütün pipeline aşamalarını sırayla çalıştırıyoruz.
if __name__ == "__main__":
    main()
