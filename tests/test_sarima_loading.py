
# MLflow kütüphanesini içe aktarıyoruz.
import mlflow

# Statsmodels modellerini yükleyebilmek için
# MLflow'un ilgili modülünü içe aktarıyoruz.
import mlflow.statsmodels


# MLflow Tracking Server adresini belirliyoruz.
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Kaydettiğimiz SARIMA modelinin URI bilgisini tanımlıyoruz.
model_uri = "models:/m-5a4236782f1342ff9cd2d7d444e60546"


# Eğitilmiş SARIMA modelini MLflow'dan yüklüyoruz.
loaded_model = mlflow.statsmodels.load_model(model_uri)


# Yüklenen modelin türünü terminale yazdırıyoruz.
print("Yüklenen model türü:", type(loaded_model))


# Yeniden eğitim yapmadan 12 aylık tahmin üretiyoruz.
predictions = loaded_model.get_forecast(steps=12).predicted_mean


# Tahmin sonuçlarını terminalde gösteriyoruz.
print("\nMLflow'dan yüklenen SARIMA tahminleri:")
print(predictions)
