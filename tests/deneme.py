
# 1. Veri işlemleri için Pandas kütüphanesini içe aktarıyoruz.
import pandas as pd

# 2. Feature Engineering fonksiyonumuzu içe aktarıyoruz.
from src.features.feature_engineering import create_features

# 3. Test için 15 aylık tarih oluşturuyoruz.
dates = pd.date_range(
    start="2024-01-01",
    periods=15,
    freq="MS"
)

# 4. Örnek satış değerlerini tanımlıyoruz.
sales = [
    100, 120, 150, 130, 140,
    160, 170, 180, 190, 200,
    210, 220, 230, 240, 250
]

# 5. Tarihleri indeks olarak kullanarak
# satış verilerinden DataFrame oluşturuyoruz.
df = pd.DataFrame(
    {"sales": sales},
    index=dates
)

# 6. Ham verimizi ekrana yazdırıyoruz.
print("\nOrijinal Satış Verisi:")
print(df)

# 7. Lag, rolling ve takvim özelliklerini oluşturuyoruz.
feature_df = create_features(df)

# 8. Yeni özellik tablomuzu ekrana yazdırıyoruz.
print("\nFeature Engineering Pipeline Sonuçları:")
print(feature_df)

# 9. Satır ve sütun sayısını kontrol ediyoruz.
print("\nVeri setinin boyutu:")
print(feature_df.shape)

# 10. Eksik değerlerin toplamını hesaplıyoruz.
print("\nToplam eksik değer sayısı:")
print(feature_df.isna().sum().sum())
