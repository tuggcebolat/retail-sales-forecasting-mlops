
# Veri tabloları oluşturmak için Pandas'ı içe aktarıyoruz.
import pandas as pd

# İlk tahmin modelimizi içe aktarıyoruz.
from src.models.baseline import seasonal_naive_forecast

# Model değerlendirme fonksiyonumuzu içe aktarıyoruz.
from src.models.evaluation import evaluate_forecast

#SARIMA modelini içe aktarıyoruz.
from src.models.sarima import sarima_forecast


# Farklı zaman dönemlerinde tahmin performansını ölçüyoruz.
def rolling_backtest(
    df: pd.DataFrame,
    initial_train_size: int = 312,
    horizon: int = 6,
    step: int = 6
) -> pd.DataFrame:

    # Eğitim verisinin en az 12 aylık olmasını sağlıyoruz.
    if initial_train_size < 12:
        raise ValueError("En az 12 aylık başlangıç eğitimi gerekli.")

    # Baseline modelimiz en fazla 12 aylık tahmin destekliyor.
    if not 1 <= horizon <= 12:
        raise ValueError("Horizon 1 ile 12 arasında olmalı.")

    # Tahmin başlangıçlarının ileri gitmesini sağlıyoruz.
    if step < 1:
        raise ValueError("Step pozitif olmalı.")

    # İlk eğitim ve test pencerelerinin sığdığını kontrol ediyoruz.
    if initial_train_size + horizon > len(df):
        raise ValueError("Veri, eğitim ve test pencereleri için yetersiz.")

    # Her test döneminin sonuçlarını saklayacağımız liste.
    results = []

    # Tahmin başlangıcını adım adım ileri taşıyoruz.
    for train_end in range(
        initial_train_size,
        len(df) - horizon + 1,
        step
    ):

        # Tahmin tarihine kadar bilinen verileri seçiyoruz.
        train_df = df.iloc[:train_end]

        # Tahmin edilecek sonraki 6 ayı seçiyoruz.
        test_df = df.iloc[train_end:train_end + horizon]

        # Yalnızca geçmiş eğitim verisiyle tahmin üretiyoruz.
        predictions = seasonal_naive_forecast(
            train_df,
            horizon=horizon
        )

        # Tahmin tarihleri test tarihleriyle eşleşmeli.
        if not predictions.index.equals(test_df.index):
            raise ValueError("Tahmin ve test tarihleri eşleşmiyor.")

        # Gerçek ve tahmin edilen satışları değerlendiriyoruz.
        metrics = evaluate_forecast(
            test_df["sales"],
            predictions
        )

        # Her test döneminin sonuçlarını kaydediyoruz.
        results.append({
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "sMAPE": metrics["sMAPE"]
        })

    # Bütün test dönemlerini tek tabloda birleştiriyoruz.
    return pd.DataFrame(results)

#İki modeli aynı veri döneminde karşılaştırmak için fonksiyon oluşturuyoruz.
def compare_models_backtest(
    df: pd.DataFrame,
    initial_train_size: int = 312,
    horizon: int = 6,
    step: int = 6
) -> pd.DataFrame:
    #Parametlerin geçerliliğini kontrol ediyoruz.
    if initial_train_size < 24 or horizon < 1 or step  < 1:
        raise ValueError("Geçersiz parametreler.")
    #Test için yeterli veri olup olmadığını kontrol ediyoruz.
    if len(df) < initial_train_size + horizon:
        raise ValueError("Veri, eğitim ve test pencereleri için yetersiz.")
    
    #Her test döneminin sonuçlarını saklayacağımız liste.
    results = []
    #İlk eğitim penceresinin bitiş konumunu belirliyoruz.
    train_end = initial_train_size
    
    #Tam bir test penceresi kaldığı sürece döngüye devam ediyoruz.
    while train_end + horizon <= len(df):
        #Eğitim verisinin başlangıçtan train_end konumuna kadar olan kısmını seçiyoruz.
        train_df = df.iloc[:train_end]
        
        #Eğitim döneminden sonraki horizon kadar olan test verisini seçiyoruz.
        test_df = df.iloc[train_end:train_end + horizon]
        
        #Test dönemindeki gerçek satışları alıyoruz.
        actual = test_df["sales"]
        
        #Seasonal Naive modelini kullanarak tahmin üretiyoruz.
        naive_predictions = seasonal_naive_forecast(
            train_df,
            horizon=horizon
        )
        
        #SARIMA modelini kullanarak tahmin üretiyoruz.
        sarima_predictions = sarima_forecast(
            train_df,
            horizon=horizon,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        
        #Seasonal Naaive tahmin hatalarını hesaplıyoruz.
        naive_metrics = evaluate_forecast(
            actual,
            naive_predictions
        )
        
        #SARIMA tahmin hatalarını hesaplıyoruz.
        sarima_metrics = evaluate_forecast(
            actual,
            sarima_predictions
        )
        
        #Seasonal Naive sonuçlarını listeye ekliyoruz.
        results.append({
            "model": "Seasonal Naive",
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            **naive_metrics
        })
        
        #SARIMA sonuçlarını aynı dönem bilgileriyle listeye ekliyoruz.
        results.append({
            "model": "SARIMA",
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            **sarima_metrics
        })
        #Eğitim penceresini step kadar ileri taşıyoruz.
        train_end += step
        
    #Bütün sonuçları tek tabloda birleştiriyoruz ve döndürüyoruz.
    return pd.DataFrame(results)