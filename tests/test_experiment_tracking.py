from src.models.experiment_tracking import log_model_metrics

# Örnek performans metrikleri oluşturuyoruz.
metrics = {
    "MAE": 100.0,
    "RMSE": 120.0,
    "sMAPE": 3.5
}

# Metrikleri MLflow sunucusuna gönderiyoruz.
log_model_metrics("test-sarima", metrics)