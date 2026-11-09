
# ========================================================
# GEREKLİ KÜTÜPHANELER VE FONKSİYONLAR
# ========================================================

# 1. Parametre kombinasyonlarını oluşturmak için kullanıyoruz.
from itertools import product

# 2. SARIMA tahmin fonksiyonumuzu içe aktarıyoruz.
from src.models.sarima import sarima_forecast

# 3. Tahmin hatalarını hesaplayan fonksiyonumuzu alıyoruz.
from src.models.evaluation import evaluate_forecast


# ========================================================
# AŞAMA 10 - SARIMA GRID SEARCH
# ========================================================

# 1. Grid Search fonksiyonumuzu tanımlıyoruz.
def sarima_grid_search(train_df):

    # 2. Deneyeceğimiz ARIMA parametrelerini belirliyoruz.
    p_values = [0, 1]
    d_values = [1]
    q_values = [0, 1]

    # 3. Bütün parametre kombinasyonlarını oluşturuyoruz.
    parameter_combinations = list(
        product(p_values, d_values, q_values)
    )

    # 4. Eğitim verisinin son 6 ayını doğrulama için ayırıyoruz.
    validation_df = train_df.iloc[-6:]

    # 5. Önceki ayları model eğitimi için kullanıyoruz.
    model_train_df = train_df.iloc[:-6]

    # 6. Her parametre kombinasyonunun sonucunu saklıyoruz.
    results = []

    # 7. Parametreleri sırayla deniyoruz.
    for order in parameter_combinations:

        try:
            # 8. SARIMA modeliyle 6 aylık tahmin üretiyoruz.
            predictions = sarima_forecast(
                train_df=model_train_df,
                horizon=6,
                order=order,
                seasonal_order=(1, 1, 1, 12)
            )

            # 9. Gerçek satışlarla tahminleri karşılaştırıyoruz.
            metrics = evaluate_forecast(
                validation_df["sales"],
                predictions
            )

            # 10. Parametreleri ve hata değerlerini kaydediyoruz.
            results.append({
                "order": order,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "sMAPE": metrics["sMAPE"]
            })

            # 11. Denemenin sonucunu ekrana yazdırıyoruz.
            print(
                f"{order} -> sMAPE: {metrics['sMAPE']:.2f}%"
            )

        except Exception as error:

            # 12. Bir kombinasyon başarısız olursa diğerlerini deniyoruz.
            print(f"{order} başarısız: {error}")

    # 13. Bütün sonuçları ana programa döndürüyoruz.
    return results


# ========================================================
# AŞAMA 11 - GRID SEARCH + ROLLING BACKTESTING
# ========================================================

# 1. Her SARIMA parametresini geçmiş dönemlerde test ediyoruz.
def sarima_grid_backtest(
    train_df,
    initial_train_size=312,
    horizon=6,
    step=6
):

    # 2. Deneyeceğimiz parametre kombinasyonlarını oluşturuyoruz.
    parameter_combinations = list(
        product([0, 1], [1], [0, 1])
    )

    # 3. Bütün kombinasyonların sonuçlarını saklayacağız.
    results = []

    # 4. Her parametre kombinasyonunu sırayla seçiyoruz.
    for order in parameter_combinations:

        # 5. Seçilen kombinasyonun dönemsel hata sonuçlarını tutuyoruz.
        errors = []

        # 6. İlk eğitim penceresinin bitişini belirliyoruz.
        train_end = initial_train_size

        # 7. Test edilen parametreyi terminalde gösteriyoruz.
        print(f"\nTest edilen SARIMA parametresi: {order}")

        # 8. Test penceresi veri sınırlarını aşmadığı sürece ilerliyoruz.
        while train_end + horizon <= len(train_df):

            # 9. O döneme kadar olan verileri eğitim için ayırıyoruz.
            model_train = train_df.iloc[:train_end]

            # 10. Sonraki 6 ayı doğrulama için ayırıyoruz.
            validation = train_df.iloc[
                train_end:train_end + horizon
            ]

            # 11. Seçilen parametrelerle tahmin üretiyoruz.
            predictions = sarima_forecast(
                train_df=model_train,
                horizon=horizon,
                order=order,
                seasonal_order=(1, 1, 1, 12)
            )

            # 12. Tahmin hatalarını hesaplıyoruz.
            metrics = evaluate_forecast(
                validation["sales"],
                predictions
            )

            # 13. Dönemin sMAPE değerini kaydediyoruz.
            errors.append(metrics["sMAPE"])

            # 14. Eğitim penceresini 6 ay ileri taşıyoruz.
            train_end += step

        # 15. Kombinasyonun ortalama hatasını hesaplıyoruz.
        average_smape = sum(errors) / len(errors)

        # 16. Parametrenin ortalama hatasını terminalde gösteriyoruz.
        print(f"Ortalama sMAPE: {average_smape:.2f}%")

        # 17. Sonucu kaydediyoruz.
        results.append({
            "order": order,
            "average_sMAPE": average_smape
        })

    # 18. Bütün kombinasyonların sonuçlarını döndürüyoruz.
    return results
