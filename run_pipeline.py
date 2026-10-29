#Veri çekme fonksiyonunu çağırıyoruz.
from src.data.ingestion import load_sales_data

#Veri doğrulama fonksiyonunu çağırıyoruz.
from src.data.validation import validate_sales_data

#Veri ön işleme fonksiyonunu çağırıyoruz.
from src.data.preprocessing import preprocess_sales_data


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
    
    
#Dosya doğrudan çalıştırıldığında main fonksiyonunu çağırıyoruz.
if __name__ == "__main__":
    main()
    