# MLflow'un standart Python model arayüzünü kullanıyoruz.
import mlflow.pyfunc

# Kaydedilmiş SARIMA modelini yüklemek için.
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


# MLflow ile uyumlu SARIMA model sınıfımızı oluşturuyoruz.
class SARIMAForecastModel(mlflow.pyfunc.PythonModel):

    # MLflow modeli yüklediğinde bu metodu çalıştırır.
    def load_context(self, context):

        # Kaydedilmiş SARIMA dosyasının yolunu alıyoruz.
        model_path = context.artifacts["sarima_model"]

        # Eğitilmiş modeli belleğe yüklüyoruz.
        self.model = SARIMAXResults.load(model_path)

#Kullanıcıdan gelen tahmin isteğini işliyoruz.
    def predict(self,context,model_input):
    #Kullanıcının kaç aylık tahmin istediğini alıyoruz.
        steps =int(model_input["steps"].iloc[0])
    
    #Geçersiz tahmin süresini engelliyoruz.
        if steps < 1:
          raise ValueError("Tahmin süresi en az 1 ay olmalıdır.")
    
    #Eğitilmiş SARIMA modeliyle tahmin üretiyoruz.
        forecast = self.model.forecast(steps=steps)
    
    #Tahminleri tablo formatında döndürüyoruz.
        return forecast.to_frame(name="predicted_sales")