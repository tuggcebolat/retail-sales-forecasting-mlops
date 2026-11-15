# MLflow bağlantısını yapılandırmak için kütüphaneyi içe aktarıyoruz.
import mlflow

# Projemizde yazdığımız alias fonksiyonunu içe aktarıyoruz.
from src.tracking.mlflow_tracking import set_champion_alias

# MLflow sunucumuzun adresini belirliyoruz.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# SARIMA Version 1 için champion alias'ını oluşturuyoruz.
set_champion_alias(
    model_name="retail-sales-sarima",
    version="1"
)