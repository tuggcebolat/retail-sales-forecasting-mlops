#Veri çekme fonksiyonunu çağırıyoruz.
from src.data.ingestion import load_sales_data

#Veri doğrulama fonksiyonunu çağırıyoruz.
from src.data.validation import validate_sales_data

#Veri ön işleme fonksiyonunu çağırıyoruz.
from src.data.preprocessing import preprocess_sales_data

from src.data.eda import analyze_sales_data, plot_sales_trend, plot_rolling_mean ,analyze_monthly_seasonality

#Pipeline ana fonksiyonunu tanımlıyoruz.
def main():
    #FRED üzerindeki satış verisini çekiyoruz.
    df = load_sales_data()
    
    #Verinin kurallarımıza uygunluğunu kontrol ediyoruz.
    validated_df = validate_sales_data(df)
    
    #Doğrulanan veriyi zaman serisine uygun hale getiriyoruz.
    processed_df = preprocess_sales_data(validated_df)
    
    #işlemlerin tamamlandığını belirten bir mesaj yazdırıyoruz.
    print("Veri çekme, doğrulama ve ön işleme işlemleri tamamlandı.")
    
    #İlk 5 satırı yazdırıyoruz.
    print(processed_df.head())  

    #Index nesnesinin Python sınıfını yazdırıyoruz.
    print(f"Index sınıfı: {type(processed_df.index)}")
    
    #İndexin türünü kontrol ediyoruz.
    print(f"İndex türü: {processed_df.index.dtype}")
    
    #Veri frekansını kontrol ediyoruz.
    print(f"Veri frekansı: {processed_df.index.freq}")
    
    #Satış verisinin özet istatistiklerini yazdırıyoruz.
    statistics = analyze_sales_data(processed_df)
    
    #İstatistikleri terminalde okunabilir bir formatta yazdırıyoruz.
    print("\nSatış Verisi İstatistikleri:")
    print(statistics)
    
    #Aylık satış trendinin görselleştirilmesini sağlıyoruz.
    plot_sales_trend(processed_df)
    
    #12 aylık hareketli ortalama ile satış trendini görselleştiriyoruz.
    plot_rolling_mean(processed_df)
    
    # Aylık mevsimsellik oranlarını hesaplıyoruz.
    monthly_profile = analyze_monthly_seasonality(processed_df)

    # Sonuçları terminalde gösteriyoruz.
    print("\nAylık mevsimsellik profili:")
    print(monthly_profile.round(3))
    
    
    
#Dosya doğrudan çalıştırıldığında main fonksiyonunu çağırıyoruz.
if __name__ == "__main__":
    main()
    