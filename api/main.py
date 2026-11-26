
# FastAPI uygulaması için gerekli kütüphaneler.
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Veri işleme ve model yükleme kütüphaneleri.
import pandas as pd
import mlflow
import os

# API metriklerini ölçmek için kullanıyoruz.
from prometheus_fastapi_instrumentator import Instrumentator

# Ortam değişkeninden MLflow adresini alıyoruz.
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

# MLflow bağlantılarını yapılandırıyoruz.
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

# Champion alias'ına sahip modeli seçiyoruz.
MODEL_URI = "models:/RetailSalesForecast@champion"

# Eğitilmiş SARIMA modelini belleğe yüklüyoruz.
model = mlflow.pyfunc.load_model(MODEL_URI)


# FastAPI uygulamamızı oluşturuyoruz.
app = FastAPI(
    title="Retail Sales Forecasting API",
    version="1.0.0"
)
# API metriklerini ölçüyor ve /metrics endpoint'ini oluşturuyoruz.
Instrumentator().instrument(app).expose(app)


# API'nin çalışıp çalışmadığını kontrol ediyoruz.
@app.get("/")
def home():
    return {"message": "Retail Sales Forecasting API is running"}


# Kullanıcının göndereceği isteğin veri yapısını tanımlıyoruz.
class PredictionRequest(BaseModel):

    # Kullanıcı 1 ile 12 ay arasında tahmin isteyebilir.
    steps: int = Field(ge=1, le=12)


# Satış tahmini endpoint'imizi oluşturuyoruz.
@app.post("/predict")
def predict(request: PredictionRequest):

    # Kullanıcının istediği ay sayısını DataFrame'e dönüştürüyoruz.
    input_data = pd.DataFrame({
        "steps": [request.steps]
    })

    # Eğitilmiş SARIMA modeliyle tahmin üretiyoruz.
    predictions = model.predict(input_data)

    # Tahmin edilen satış değerlerini Python listesine çeviriyoruz.
    forecast_values = predictions["predicted_sales"].tolist()

    # Tahminlerin ait olduğu tarihleri listeye dönüştürüyoruz.
    forecast_dates = predictions.index.strftime("%Y-%m-%d").tolist()

    # Tarihleri ve satış tahminlerini aynı listede birleştiriyoruz.
    forecasts = []

    # Her tarih ve tahmin değerini eşleştiriyoruz.
    for date, value in zip(forecast_dates, forecast_values):

        # Her ay için bir sözlük oluşturuyoruz.
        forecasts.append({
            "date": date,
            "predicted_sales": float(value)
        })

    # Tahmin sonuçlarını JSON yanıtı olarak döndürüyoruz.
    return {
        "forecast_months": request.steps,
        "model_name": "SARIMA",
        "forecasts": forecasts
    }
