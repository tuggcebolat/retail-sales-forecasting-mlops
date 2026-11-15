
# MLflow kütüphanesini içe aktarıyoruz.
import mlflow
# MLflow istemcisini içe aktarıyoruz.
from mlflow import MlflowClient

# MLflow sunucusu ve experiment ayarlarını yapılandırıyoruz.
def setup_mlflow():

    # Yerel MLflow Tracking Server adresini belirliyoruz.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Deneylerimizin kaydedileceği experiment'ı seçiyoruz.
    mlflow.set_experiment("retail-sales-forecasting")


# Model deneylerini parametre, metrik ve etiketlerle kaydediyoruz.
def log_model_experiment(
    model_name,
    params,
    metrics,
    metadata
):

    # Model adına göre yeni bir MLflow run oluşturuyoruz.
    with mlflow.start_run(
        run_name=f"{model_name}_Backtesting"
    ):

        # Model adını parametre olarak kaydediyoruz.
        mlflow.log_param("model", model_name)

        # Modelin hiperparametrelerini kaydediyoruz.
        mlflow.log_params({
            key: str(value)
            for key, value in params.items()
        })

        # Backtesting performans metriklerini kaydediyoruz.
        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
        })

        # Deneyin açıklayıcı bilgilerini etiket olarak kaydediyoruz.
        mlflow.set_tags({
            key: str(value)
            for key, value in metadata.items()
        })

        # Başarılı kayıt işlemini terminale yazdırıyoruz.
        print(f"{model_name} deneyi MLflow'a kaydedildi.")
# Eğitilmiş SARIMA modelini MLflow'a kaydeden fonksiyon.
def save_sarima_model(fitted_model):

    # Modeli kaydetmek için yeni bir MLflow çalıştırması açıyoruz.
    with mlflow.start_run(run_name="SARIMA_Model_Save"):

        # SARIMA modelini MLflow artifact deposuna kaydediyoruz.
        model_info = mlflow.statsmodels.log_model(
            statsmodels_model=fitted_model,
            name="sarima_model"
        )

        # Kaydedilen modelin MLflow adresini yazdırıyoruz.
        print("SARIMA modeli kaydedildi.")
        print("Model URI:", model_info.model_uri)

        # Model bilgilerini diğer fonksiyonlarda kullanmak
        # üzere geri döndürüyoruz.
        return model_info

def register_sarima_model(model_uri):

    # Eğitilmiş modeli MLflow Model Registry'ye kaydediyoruz.
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="retail-sales-sarima"
    )

    # Oluşturulan model sürümünü gösteriyoruz.
    print("Model adı:", registered_model.name)
    print("Model sürümü:", registered_model.version)

    return registered_model



# MLflow Model Registry işlemleri için istemciyi içe aktarıyoruz.
from mlflow import MlflowClient


# Model adı ve sürüm numarasını parametre olarak alıyoruz.
def set_champion_alias(model_name, version):

    # MLflow istemcisini oluşturuyoruz.
    client = MlflowClient()

    # Belirtilen model sürümüne champion alias'ını atıyoruz.
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version
    )

    # Atanan alias'ı tekrar sorgulayarak doğruluyoruz.
    registered_version = client.get_model_version_by_alias(
        name=model_name,
        alias="champion"
    )

    # Alias'ın doğru sürüme ait olduğunu kontrol ediyoruz.
    assert str(registered_version.version) == str(version)

    # İşlem başarılıysa bilgi veriyoruz.
    print(f"{model_name} Version {version} champion olarak ayarlandı.")