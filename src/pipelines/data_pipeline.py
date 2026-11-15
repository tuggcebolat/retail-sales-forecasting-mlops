
# FRED veri kaynağından satış verilerini çekmek için
# mevcut ingestion fonksiyonumuzu içe aktarıyoruz.
from src.data.ingestion import load_sales_data

# Satış verilerinin kalitesini kontrol etmek için
# mevcut validation fonksiyonumuzu içe aktarıyoruz.
from src.data.validation import validate_sales_data

# Verileri zaman serisi formatına dönüştürmek için
# preprocessing fonksiyonumuzu içe aktarıyoruz.
from src.data.preprocessing import preprocess_sales_data


# Veri hazırlama işlemlerini yöneten fonksiyonumuzu tanımlıyoruz.
def prepare_sales_data():

    # AŞAMA 1: DATA INGESTION
    # FRED üzerinden aylık satış verilerini çekiyoruz.
    df = load_sales_data()

    # AŞAMA 2: DATA VALIDATION
    # Eksik, negatif veya tekrar eden verileri kontrol ediyoruz.
    validated_df = validate_sales_data(df)

    # AŞAMA 3: DATA PREPROCESSING
    # Tarih indeksini ve aylık frekansı hazırlıyoruz.
    processed_df = preprocess_sales_data(validated_df)

    # İşlenmiş verinin satır sayısını ekrana yazdırıyoruz.
    print("Veri hazırlama işlemi tamamlandı.")
    print(f"Toplam gözlem sayısı: {len(processed_df)}")

    # Hazırlanan DataFrame'i diğer modüllerin kullanması
    # için geri döndürüyoruz.
    return processed_df
