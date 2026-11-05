import pandas as pd

#Geçen yılın aynı ayına göre satış değişimini hesaplıyoruz.
def seasonal_naive_forecast(
    train_df:pd.DataFrame,
    horizon:int =12
) -> pd.Series:
    
    #Eğitim verisinin en az 12 aylık veri içerdiğini doğruluyoruz.
    if len(train_df) < 12:
        raise ValueError("Eğitim verisi en az 12 aylık veri içermelidir.")
    
    #Tahmin ufkunun geçerli olduğunu doğruluyoruz.
    if not 1 <= horizon <= 12:
        raise ValueError("Tahmin ufku 1 ile 12 arasında olmalıdır.")
    
    #Eğitim verisinin son 12 ayını alıyoruz.
    last_year_sales =train_df["sales"].iloc[-12:]
    
    #İstenen ay sayısı kadar geçmiş satış seçiyoruz.
    predictions = last_year_sales.iloc[:horizon].copy()
    
    #Tahmin edilecek gelecek ayların  tarihlerini oluşturuyoruz.
    future_dates = pd.date_range(
        start=train_df.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS")
    
    #Tahminlerin index'ini gelecek tarihlerle değiştiriyoruz.
    predictions.index = future_dates
    
    #Tahmin verisini döndürüyoruz.
    return predictions