from src.pipelines.data_pipeline import prepare_sales_data
from src.models.model_training import train_selected_model
from src.models.model_persistence import save_selected_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


# Verilerimizi hazırlıyoruz.
df = prepare_sales_data()

# SARIMA modelimizi eğitiyoruz.
model = train_selected_model(df, "SARIMA")

# Eğitilmiş modeli kaydediyoruz.
save_selected_model(model, "SARIMA", "sarima.pkl")


# Kaydedilen modeli yük­lüyoruz.
loaded_model = SARIMAXResults.load("sarima.pkl")

# Gelecek 6 aylık satışları tahmin ediyoruz.
forecast = loaded_model.forecast(steps=6)

# Tahmin sonuçlarını görüntülüyoruz.
print(forecast)