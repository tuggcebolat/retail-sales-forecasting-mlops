# MLflow kütüphanesini içe aktarıyoruz.
import mlflow

# Statsmodels modellerini yüklemek için gerekli modülü içe aktarıyoruz.
import mlflow.statsmodels

# MLflow sunucumuzun adresini belirliyoruz.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Champion olarak belirlenen model sürümünü kullanıyoruz.
model_uri = "models:/retail-sales-sarima@champion"

# Modeli Registry üzerinden yüklüyoruz.
loaded_model = mlflow.statsmodels.load_model(model_uri)

# Yüklenen modelin türünü görüntülüyoruz.
print("Yüklenen model türü:", type(loaded_model))

# Yeniden eğitim yapmadan 6 aylık tahmin üretiyoruz.
predictions = loaded_model.get_forecast(
    steps=6
).predicted_mean

# Tahminleri terminalde görüntülüyoruz.
print("\nModel Registry üzerinden üretilen tahminler:")
print(predictions)