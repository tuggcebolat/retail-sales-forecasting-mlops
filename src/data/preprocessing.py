
import pandas as pd


def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:

    # Orijinal DataFrame'in değişmesini önlemek için kopyalıyoruz.
    processed_df = df.copy()

    # Tarih sütununu DataFrame'in index'i yapıyoruz.
    processed_df = processed_df.set_index("observation_date")

    # Verileri tarih sırasına göre düzenliyoruz.
    processed_df = processed_df.sort_index()

    # Verilerin aylık frekansını tanımlıyoruz.
    processed_df = processed_df.asfreq("MS")

    # İşlenmiş DataFrame'i döndürüyoruz.
    return processed_df
