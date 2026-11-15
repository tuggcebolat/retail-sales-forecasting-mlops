
# Veri hazırlama fonksiyonumuzu içe aktarıyoruz.
from src.pipelines.data_pipeline import prepare_sales_data


# Veri hazırlama pipeline'ını çalıştırıyoruz.
processed_df = prepare_sales_data()

# İlk 5 satırı görüntülüyoruz.
print("\nİlk 5 gözlem:")
print(processed_df.head())

# Tarih indeksinin frekansını kontrol ediyoruz.
print("\nVeri frekansı:", processed_df.index.freq)

# Veri setinin boş olmadığını doğruluyoruz.
assert not processed_df.empty, "Veri seti boş!"

# Satış sütununun bulunduğunu doğruluyoruz.
assert "sales" in processed_df.columns, "Sales sütunu bulunamadı!"

# Testin başarılı olduğunu bildiriyoruz.
print("\nData Pipeline testi başarılı!")
