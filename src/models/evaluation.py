#Sayısal işlemler için NumPy kütüphanesini içe aktarıyoruz.
import numpy as np

#Hata metriklerini hesaplamak için scikit-learn kütüphanesinden gerekli fonksiyonları içe aktarıyoruz.
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Modelin tahmin performansını değerlendiren bir fonksiyon tanımlıyoruz.
def evaluate_forecast(actual,predicted):
    #Gerçek değerleri numpy dizisine çeviriyoruz.
    actual = np.asarray(actual, dtype=np.float64)
    
    #Tahmin edilen değerleri numpy dizisine çeviriyoruz.
    predicted = np.asarray(predicted, dtype=np.float64)
    
    #Gözlem sayılarının eşit olduğunu kontrol ediyoruz.
    if actual.shape != predicted.shape:
        raise ValueError("Gerçek ve tahmin edilen değerlerin boyutları eşit olmalıdır.")
    
    #Ortalama Mutlak Hata (MAE) hesaplıyoruz.
    mae = mean_absolute_error(actual, predicted)
    
    #Ortalama Kare Hata (MSE) hesaplıyoruz.
    mse = np.sqrt(mean_squared_error(actual, predicted))
    
    # sMAPE hesaplamasında kullanılacak paydayı oluşturuyoruz.
    denominator = np.abs(actual) + np.abs(predicted)
    
    # Her iki değer de sıfırsa katkıyı sıfır kabul ediyoruz.
    smape_values =np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(actual, dtype=np.float64),
        where=denominator != 0
        
    )
    #Ortalama Mutlak Yüzde Hata (sMAPE) hesaplıyoruz.
    smape = np.mean(smape_values) * 100
    
    #Metrikleri bir sözlük olarak döndürüyoruz.
    return {
        "MAE": mae, 
        "RMSE": mse,
        "sMAPE": smape
    }