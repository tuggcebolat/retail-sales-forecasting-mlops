import pandas  as  pd

#Doğrulanmış satış verisini zaman serisine hazırlıyoruz.

def preprocess_sales_data(df:pd.DataFrame) -> pd.DataFrame:
    
    #Orijinal veri çerçevesini kopyalıyoruz.
    processed_df = df.copy()
    
    #Tarih sütununu indeks olarak ayarlıyoruz.
    processed_df.set_index("observation_date", inplace=True)
    
    #Tarih indeksini artan sıralı hale getiriyoruz.
    processed_df.sort_index(inplace=True)
    
    #Veri setinin aylık başlangıç frekansına sahip olduğunu doğruluyoruz.
    processed_df = processed_df.asfreq("MS")
    
    #Ön işlenmiş veri çerçevesini döndürüyoruz.
    return processed_df
