#veri analizi için pandas kütüphanesini içe aktarıyoruz.
import pandas as pd

#veri doğrulama işlemlerini bir fonksiyon içinde tanımlıyoruz.
def validate_sales_data(df:pd.DataFrame) -> pd.DataFrame:
    #beklediğimiz sütunları tanımlıyoruz.
    required_columns= {"observation_date", "sales"}
    
    #veri setinde bulunmayan sütunları belirliyoruz.
    missing_columns = required_columns - set(df.columns)
    
    #eksik sütunlar varsa işlemi durduruyoruz
    if missing_columns:
        raise ValueError(f"eksik sütunlar var: {missing_columns}")
    
    #boş veri setini kabul etmiyoruz.
    if df.empty:
        raise ValueError("veri seti boş")
    
    #Tarih ve satış sütunlarında boş değerleri kontrol ediyoruz.
    if df[["observation_date", "sales"]].isnull().any().any():
        raise ValueError("Tarih veya satış sütununda boş değerler var")
    
    #Tarih sütunundaki değerlerin datetime tipinde olduğunu kontrol ediyoruz.
    if not pd.api.types.is_datetime64_any_dtype(
        df["observation_date"]
        ):
        raise TypeError("Tarih sütunu datetime tipinde değil")
    
    #Satış sütunundaki değerlerin sayısal tipte olduğunu kontrol ediyoruz.
    if not pd.api.types.is_numeric_dtype(df["sales"]):
        raise TypeError("Satış sütunu sayısal tipte değil")
    
    #Negatif satış değerlerini kontrol ediyoruz.
    if (df["sales"]<0).any():
        raise ValueError("Satış sütununda negatif değerler var")
    
    #Tekrarlanan tarihleri kontrol ediyoruz.
    if df["observation_date"].duplicated().any():
        raise ValueError("Tarih sütununda tekrarlanan değerler var")
    
    #Tarihleri kronolojik sıraya göre sıralıyoruz.
    df.sort_values(by="observation_date").copy()
    
    #Beklenen aylık tarih aralığını kontrol ediyoruz.
    expected_dates =pd.date_range(
        start=df["observation_date"].min(),
        end=df["observation_date"].max(),
        freq="MS"
    )
    
    #Eksik veya düzensiz ayları kontrol ediyoruz.
    actual_dates =pd.DatetimeIndex(df["observation_date"])
    if not actual_dates.equals(expected_dates):
        raise ValueError("Tarih sütununda eksik veya düzensiz aylar var")
    
    #Kontrol edilen veri çerçevesini döndürüyoruz.
    return df.reset_index(drop=True)
    