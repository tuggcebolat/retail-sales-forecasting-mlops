
# Veri işleme işlemleri için Pandas kütüphanesini içe aktarıyoruz.
import pandas as pd


# ============================================================
# 1. LAG FEATURES (GECİKMELİ ÖZELLİKLER)
# ============================================================

# Geçmiş satışlardan gecikmeli özellikler oluşturacak
# fonksiyonumuzu tanımlıyoruz.
def create_lag_features(df):

    # Orijinal veri setini değiştirmemek için kopyasını oluşturuyoruz.
    featured_df = df.copy()

    # Bir önceki ayın satışlarını yeni sütuna ekliyoruz.
    featured_df["lag_1"] = featured_df["sales"].shift(1)

    # Üç ay önceki satışları yeni sütuna ekliyoruz.
    featured_df["lag_3"] = featured_df["sales"].shift(3)

    # Geçen yılın aynı ayına ait satışları yeni sütuna ekliyoruz.
    featured_df["lag_12"] = featured_df["sales"].shift(12)

    # Yeni özellikler eklenmiş veri çerçevesini döndürüyoruz.
    return featured_df


# ============================================================
# 2. ROLLING FEATURES (HAREKETLİ ORTALAMA ÖZELLİKLERİ)
# ============================================================

# Geçmiş satışların hareketli ortalamalarını hesaplayan
# fonksiyonumuzu tanımlıyoruz.
def create_rolling_mean_features(df):

    # Orijinal veri setini değiştirmemek için kopyasını oluşturuyoruz.
    featured_df = df.copy()

    # Mevcut ayın satışını kullanmamak için
    # satış serisini bir dönem kaydırıyoruz.
    previous_sales = featured_df["sales"].shift(1)

    # Önceki 3 ayın ortalama satışını hesaplıyoruz.
    featured_df["rolling_mean_3"] = (
        previous_sales.rolling(window=3).mean()
    )

    # Önceki 6 ayın ortalama satışını hesaplıyoruz.
    featured_df["rolling_mean_6"] = (
        previous_sales.rolling(window=6).mean()
    )

    # Yeni özellikler eklenmiş veri çerçevesini döndürüyoruz.
    return featured_df


# ============================================================
# 3. CALENDAR FEATURES (TAKVİM ÖZELLİKLERİ)
# ============================================================

# Tarih indeksinden takvim özellikleri oluşturacak
# fonksiyonumuzu tanımlıyoruz.
def create_calendar_features(df):

    # Orijinal veri setini değiştirmemek için kopyasını oluşturuyoruz.
    featured_df = df.copy()

    # Tarih indeksinden ay bilgisini çıkarıyoruz.
    featured_df["month"] = featured_df.index.month

    # Tarih indeksinden çeyrek dönem bilgisini çıkarıyoruz.
    featured_df["quarter"] = featured_df.index.quarter

    # Tarih indeksinden yıl bilgisini çıkarıyoruz.
    featured_df["year"] = featured_df.index.year

    # Yeni özellikler eklenmiş veri çerçevesini döndürüyoruz.
    return featured_df

# Lag, rolling ve takvim özelliklerini bir arada
# oluşturacak fonksiyonumuzu tanımlıyoruz.
def create_features(df):

    # 1. Önce gecikmeli satış özelliklerini oluşturuyoruz.
    featured_df = create_lag_features(df)

    # 2. Hareketli ortalama özelliklerini ekliyoruz.
    featured_df = create_rolling_mean_features(featured_df)

    # 3. Tarih indeksinden takvim özelliklerini ekliyoruz.
    featured_df = create_calendar_features(featured_df)

    # 4. Lag ve rolling işlemleri nedeniyle oluşan
    # eksik değerleri içeren ilk satırları kaldırıyoruz.
    featured_df = featured_df.dropna()

    # 5. Özellikleri tamamlanmış veri setini döndürüyoruz.
    return featured_df
