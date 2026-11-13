
# MLflow kütüphanesini içe aktarıyoruz.
import mlflow


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
