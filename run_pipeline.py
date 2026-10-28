#Veri çekme fonksiyonunu çağırıyoruz.
from src.data.ingestion import load_sales_data

#Veri doğrulama fonksiyonunu çağırıyoruz.
from src.data.validation import validate_sales_data

#Pipeline ana fonksiyonunu tanımlıyoruz.
def main():
    #FRED üzerindeki satış verisini çekiyoruz.
    df = load_sales_data()
    
    #Verinin kurallarımıza uygunluğunu kontrol ediyoruz.
    validated_df = validate_sales_data(df)
    
    #Başarılı bir şekilde doğrulanan veri çerçevesini yazdırıyoruz.
    print("Veri doğrulama başarılı.")
    
    #İlk 5 satırı yazdırıyoruz.
    print(validated_df.head())
    
#Dosya doğrudan çalıştırıldığında main fonksiyonunu çağırıyoruz.
if __name__ == "__main__":
    main()
    