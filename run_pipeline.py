
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

# 7. SARIMA tahmin modelimizi içe aktarıyoruz.
from src.models.sarima import sarima_forecast

# 8. MAE, RMSE ve sMAPE metriklerini hesaplayan fonksiyonu çağırıyoruz.
from src.models.evaluation import evaluate_forecast

# 9. Rolling-Origin Backtesting fonksiyonlarımızı içe aktarıyoruz.
from src.models.backtesting import (
    rolling_backtest,
    compare_models_backtest,
)

# 10. Grid Search ve Grid Backtesting fonksiyonlarını içe aktarıyoruz.
from src.models.grid_search import (
    sarima_grid_search,
    sarima_grid_backtest,
)

# 11. XGBoost için geçmiş satışlardan model özelliklerini oluşturuyoruz.
from src.features.feature_engineering import create_features

# 12. XGBoost regresyon modelinin eğitim fonksiyonunu içe aktarıyoruz.
from src.models.xgboost_model import (
    train_xgboost_model,
    recursive_forecast_xgboost
)

# 11. Projemizin bütün aşamalarını yöneten ana fonksiyonu tanımlıyoruz.
def main():

    # ========================================================
    # AŞAMA 1 - DATA INGESTION
    # ========================================================

    # 12. FRED veri kaynağından aylık satış verilerini çekiyoruz.
    df = load_sales_data()

    # ========================================================
    # AŞAMA 2 - DATA VALIDATION
    # ========================================================

    # 13. Eksik değer, negatif satış ve tekrar eden tarih gibi
    # veri kalitesi kurallarını kontrol ediyoruz.
    validated_df = validate_sales_data(df)

    # ========================================================
    # AŞAMA 3 - DATA PREPROCESSING
    # ========================================================

    # 14. Tarih sütununu DatetimeIndex yapısına dönüştürüyor,
    # verileri sıralıyor ve aylık frekansı tanımlıyoruz.
    processed_df = preprocess_sales_data(validated_df)

    # 15. İlk üç aşamanın tamamlandığını terminale bildiriyoruz.
    print("Veri çekme, doğrulama ve ön işleme işlemleri tamamlandı.")

    # 16. İşlenmiş verinin ilk 5 satırını gösteriyoruz.
    print(processed_df.head())

    # 17. Index'in Python sınıfını kontrol ediyoruz.
    print(f"Index sınıfı: {type(processed_df.index)}")

    # 18. Index'in veri tipini kontrol ediyoruz.
    print(f"Index türü: {processed_df.index.dtype}")

    # 19. Verinin aylık frekansa sahip olduğunu doğruluyoruz.
    print(f"Veri frekansı: {processed_df.index.freq}")

    # ========================================================
    # AŞAMA 4 - EXPLORATORY DATA ANALYSIS (EDA)
    # ========================================================

    # 20. Satış verisinin özet istatistiklerini hesaplıyoruz.
    statistics = analyze_sales_data(processed_df)

    # 21. Ortalama, standart sapma, minimum ve maksimum gibi
    # istatistikleri terminalde gösteriyoruz.
    print("\nSatış Verisi İstatistikleri:")
    print(statistics)

    # 22. Satışların zaman içindeki değişimini görselleştiriyoruz.
    plot_sales_trend(processed_df)

    # 23. 12 aylık hareketli ortalamayı görselleştirerek
    # uzun dönemli satış trendini inceliyoruz.
    plot_rolling_mean(processed_df)

    # 24. Her ayın mevsimsel satış profilini hesaplıyoruz.
    monthly_profile = analyze_monthly_seasonality(processed_df)

    # 25. Aylık mevsimsellik oranlarını terminalde gösteriyoruz.
    print("\nAylık mevsimsellik profili:")
    print(monthly_profile.round(3))

    # 26. Satış serisini trend, mevsimsellik ve kalıntı
    # bileşenlerine ayırarak görselleştiriyoruz.
    plot_seasonal_decomposition(processed_df)

    # ========================================================
    # AŞAMA 5 - TRAIN / TEST SPLIT
    # ========================================================

    # 27. Son 12 aylık gerçek satışları test setine ayırıyoruz.
    test_df = processed_df.iloc[-12:]

    # 28. Son 12 ay dışındaki geçmiş verileri eğitim seti
    # olarak kullanıyoruz.
    train_df = processed_df.iloc[:-12]

    # 29. Eğitim ve test dönemlerini terminalde gösteriyoruz.
    print("\nEğitim ve Test Dönemleri:")
    print(f"Eğitim başlangıcı: {train_df.index.min()}")
    print(f"Eğitim bitişi: {train_df.index.max()}")
    print(f"Test başlangıcı: {test_df.index.min()}")
    print(f"Test bitişi: {test_df.index.max()}")

    # ========================================================
    # AŞAMA 6 - SEASONAL NAIVE MODEL
    # ========================================================

    # 30. Geçen yılın aynı ayındaki satışları kullanarak
    # test dönemindeki 12 ay için tahmin üretiyoruz.
    naive_predictions = seasonal_naive_forecast(
        train_df,
        horizon=12
    )

    # 31. Seasonal Naive tahminlerini gerçek satışlarla
    # karşılaştırarak hata metriklerini hesaplıyoruz.
    naive_metrics = evaluate_forecast(
        test_df["sales"],
        naive_predictions
    )

    # 32. Seasonal Naive modelinin metriklerini gösteriyoruz.
    print("\nSeasonal Naive Model Performansı:")

    for metric, value in naive_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ========================================================
    # AŞAMA 7 - SARIMA MODEL
    # ========================================================

    # 33. SARIMA modelini yalnızca eğitim verisiyle eğitiyoruz.
    # order=(p,d,q) normal ARIMA parametrelerini belirtir.
    # seasonal_order=(P,D,Q,s) mevsimsel parametreleri belirtir.
    sarima_predictions = sarima_forecast(
        train_df=train_df,
        horizon=12,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    # 34. SARIMA tahminlerini gerçek satışlarla karşılaştırarak
    # MAE, RMSE ve sMAPE metriklerini hesaplıyoruz.
    sarima_metrics = evaluate_forecast(
        test_df["sales"],
        sarima_predictions
    )

    # 35. SARIMA modelinin performans metriklerini gösteriyoruz.
    print("\nSARIMA Model Performansı:")

    for metric, value in sarima_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ========================================================
    # AŞAMA 8 - MODEL COMPARISON
    # ========================================================

    # 36. Gerçek satışları ve iki modelin tahminlerini
    # aynı DataFrame içerisinde birleştiriyoruz.
    comparison = pd.DataFrame({
        "Actual Sales": test_df["sales"],
        "Seasonal Naive": naive_predictions,
        "SARIMA": sarima_predictions
    })

    # 37. Aylık tahmin karşılaştırmasını terminalde gösteriyoruz.
    print("\nGerçek ve Tahmin Edilen Satışlar:")
    print(comparison)

    # 38. İki modelin hata metriklerini karşılaştırmak için
    # yeni bir DataFrame oluşturuyoruz.
    metrics_comparison = pd.DataFrame({
        "Seasonal Naive": naive_metrics,
        "SARIMA": sarima_metrics
    })

    # 39. Modellerin performanslarını yan yana gösteriyoruz.
    print("\nModel Performans Karşılaştırması:")
    print(metrics_comparison.round(4))

    # ========================================================
    # AŞAMA 9 - ROLLING-ORIGIN BACKTESTING
    # ========================================================

    # 40. Seasonal Naive modelini farklı geçmiş dönemlerde
    # 6 aylık tahmin pencereleriyle değerlendiriyoruz.
    backtest_results = rolling_backtest(
        train_df,
        initial_train_size=312,
        horizon=6,
        step=6
    )

    # 41. Her test döneminin MAE, RMSE ve sMAPE
    # sonuçlarını terminalde gösteriyoruz.
    print("\nRolling Backtesting Sonuçları:")
    print(backtest_results.to_string(index=False))

    # 42. Tüm backtesting dönemlerinin ortalama
    # performans metriklerini hesaplayıp gösteriyoruz.
    print("\nOrtalama Backtesting Performansı:")

    print(
        backtest_results[["MAE", "RMSE", "sMAPE"]]
        .mean()
        .round(2)
    )

    # 43. İki modelin Rolling-Origin Backtesting
    # sonuçlarını karşılaştırıyoruz.
    model_backtest_results = compare_models_backtest(
        train_df,
        initial_train_size=312,
        horizon=6,
        step=6
    )

    # 44. Her modelin bütün dönemlerdeki sonuçlarını gösteriyoruz.
    print("\nModellerin Rolling Backtesting Karşılaştırması:")
    print(model_backtest_results.to_string(index=False))

    # 45. Model adına göre sonuçları grupluyoruz.
    grouped_results = model_backtest_results.groupby("model")

    # 46. Her modelin ortalama hata metriklerini hesaplıyoruz.
    average_metrics = grouped_results[["MAE", "RMSE", "sMAPE"]].mean()

    # 47. Her modelin ortalama performansını terminalde gösteriyoruz.
    print("\nModellerin Ortalama Performans Metrikleri:")
    print(average_metrics.round(2))

    # 48. Ortalama sMAPE değeri en düşük olan modeli belirliyoruz.
    best_model = average_metrics["sMAPE"].idxmin()

    # 49. En düşük sMAPE değerine sahip modeli terminalde bildiriyoruz.
    print(f"\nEn iyi model: {best_model} (En düşük sMAPE)")

    # ========================================================
    # AŞAMA 10 - SARIMA GRID SEARCH
    # ========================================================

    # 50. Eğitim verisi üzerinde SARIMA parametre
    # kombinasyonlarını deniyoruz.
    grid_results = sarima_grid_search(train_df)

    # 51. Grid Search sonuçlarını kontrol ediyoruz.
    if grid_results:

        # 52. Sözlük listesini Pandas DataFrame'e dönüştürüyoruz.
        grid_results_df = pd.DataFrame(grid_results)

        # 53. Kombinasyonları en düşük sMAPE ilk sırada
        # olacak şekilde sıralıyoruz.
        grid_results_df = grid_results_df.sort_values(
            by="sMAPE",
            ascending=True
        )

        # 54. Performans sonuçlarını okunabilir bir tablo
        # olarak gösteriyoruz.
        print("\nSARIMA Grid Search Performans Tablosu:")
        print(grid_results_df.round(2).to_string(index=False))

        # 55. İlk satır, en düşük sMAPE değerine sahip kombinasyondur.
        best_result = grid_results_df.iloc[0]

        # 56. En iyi doğrulama parametrelerini ve sMAPE
        # değerini gösteriyoruz.
        print("\nEn iyi SARIMA parametreleri (tek doğrulama dönemi):")
        print(f"order: {best_result['order']}")
        print(f"sMAPE: {best_result['sMAPE']:.2f}%")

    else:

        # 57. Başarılı bir kombinasyon yoksa tablo oluşturmuyoruz.
        print("\nBaşarılı SARIMA kombinasyonu bulunamadı.")

    # ========================================================
    # AŞAMA 12 - SARIMA GRID BACKTESTING SONUÇLARI
    # ========================================================

    # 58. Dört SARIMA parametre kombinasyonunu geçmiş
    # dönemlerde Rolling-Origin Backtesting ile test ediyoruz.
    grid_backtest_results = sarima_grid_backtest(
        train_df=train_df,
        initial_train_size=312,
        horizon=6,
        step=6
    )

    # 59. Fonksiyondan başarılı sonuç dönüp dönmediğini kontrol ediyoruz.
    if grid_backtest_results:

        # 60. Sonuçları Pandas DataFrame yapısına dönüştürüyoruz.
        grid_backtest_df = pd.DataFrame(grid_backtest_results)

        # 61. Parametreleri ortalama sMAPE değerine göre
        # küçükten büyüğe sıralıyoruz.
        grid_backtest_df = grid_backtest_df.sort_values(
            by="average_sMAPE",
            ascending=True
        )

        # 62. Her parametrenin ortalama hata oranını gösteriyoruz.
        print("\nSARIMA Grid Backtesting Sonuçları:")
        print(grid_backtest_df.round(2).to_string(index=False))

        # 63. İlk satırdaki en düşük hata değerine sahip
        # SARIMA parametre kombinasyonunu seçiyoruz.
        best_backtest_result = grid_backtest_df.iloc[0]

        # 64. En başarılı SARIMA parametresini ekrana yazdırıyoruz.
        print("\nEn İyi SARIMA Parametresi (Rolling Backtesting):")
        print(f"Order: {best_backtest_result['order']}")

        # 65. Seçilen parametrenin ortalama sMAPE değerini gösteriyoruz.
        print(
            f"Ortalama sMAPE: "
            f"{best_backtest_result['average_sMAPE']:.2f}%"
        )

    else:

        # 66. Sonuç üretilemezse kullanıcıya bilgi veriyoruz.
        print("\nGrid Backtesting sonucu bulunamadı.")
    
    # ========================================================
    # Aşama 13 - Özellik Mühendisliği (Feature Engineering)
    # ========================================================
    
    #1. Eğitim dönemindeki geçmiş satışlardan
    #lag, rolling ve takvim özelliklerini oluşturuyoruz.
    feature_train_df = create_features(train_df)
    
    # 2.Hedef  satış sütununu çıkararak
    #modelin kullanacağı girdi özelliklerini oluşturuyoruz.
    X_train = feature_train_df.drop(columns=["sales"])
    
    #3. modelin öğrenmeye çalışacağı
    #gerçek satış değerlerini belirliyoruz.
    y_train = feature_train_df["sales"]
    
    #Oluşturlan özelliklerin ilk 5 satırını terminalde gösteriyoruz.
    print("\nXGBoost Eğitim Özellikleri:")
    print(X_train.head())
    
    #5.Girdi ve hedef değişkenlerin boyutlarını terminalde gösteriyoruz.
    print("\nXGBoost Eğitim Özellikleri Boyutu:")
    print(f"Girdi Özellikleri: {X_train.shape}")
    print(f"Hedef Değişken: {y_train.shape}")
    
    #6 Eğitim özelliklerinde eksik değer olup olmadığını kontrol ediyoruz.
    print("\nEğitim Özelliklerinde Eksik Değer Kontrolü:")
    print(X_train.isna().sum().sum())
    
    # ========================================================
    # AŞAMA 14 - XGBOOST MODEL TRAINING
    # ========================================================

    # Hazırladığımız eğitim verileriyle XGBoost'u eğitiyoruz.
    xgboost_model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train
    )

    # Eğitimin tamamlandığını bildiriyoruz.
    print("\nXGBoost modeli başarıyla eğitildi.")

    # Modelin türünü kontrol ediyoruz.
    print("Model türü:", type(xgboost_model))
    
    # Eğitilmiş XGBoost modeliyle gelecek 12 ayı tahmin ediyoruz.
    xgboost_forecast =recursive_forecast_xgboost(
        model=xgboost_model,
        train_df=train_df,
        horizon=12
    )
    #Tahmin sonuçlarını yazdırıyoruz.
    print("\nXGBoost Recursive Forecasting Sonuçları:")
    print(xgboost_forecast)
    
# ========================================================
# PIPELINE BAŞLATMA
# ========================================================

# Dosya doğrudan çalıştırıldığında ana pipeline'ı başlatıyoruz.
if __name__ == "__main__":
    main()
