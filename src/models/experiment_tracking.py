import mlflow

def setup_experiment():

    # MLflow sunucusunun adresini belirliyoruz.
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Deneyimizi oluşturuyor veya mevcut deneyi seçiyoruz.
    mlflow.set_experiment("retail-sales-forecasting")

def log_model_metrics(model_name, metrics):

    # İlgili MLflow deneyini seçiyoruz.
    setup_experiment()

    # Modele özel bir deney çalışması başlatıyoruz.
    with mlflow.start_run(run_name=model_name):

        # Modelin ismini kaydediyoruz.
        mlflow.log_param("model_name", model_name)

        # Performans metriklerini kaydediyoruz.
        mlflow.log_metrics(metrics)

def log_all_models(summary):

    # Her modelin performans sonuçlarını dolaşıyoruz.
    for _, row in summary.iterrows():

        # Modelin performans metriklerini alıyoruz.
        metrics = {
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "sMAPE": float(row["sMAPE"])
        }

        # Sonuçları MLflow'a kaydediyoruz.
        log_model_metrics(row["model"], metrics)