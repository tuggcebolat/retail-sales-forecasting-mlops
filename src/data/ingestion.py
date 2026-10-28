# HTTP üzerinden CSV verisini okumak için
# pandas kütüphanesini içe aktarıyoruz.
import pandas as pd

# veri kaynağının URL'sini tanımlıyoruz.
FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=MRTSSM4541USN"
)
#Veri çekme işlemini bir fonksiyon içinde tanımlıyoruz.
def load_sales_data() -> pd.DataFrame:
      # FRED üzerindeki CSV verisini okuyoruz.
      df=pd.read_csv(FRED_URL)
      # Tarih sütununu datetime tipine çeviriyoruz.
      df['observation_date'] = pd.to_datetime(df['observation_date'])
      #Satış sütununa anlaşılır bir isim veriyoruz.
      df.rename(columns={'MRTSSM4541USN': 'sales'}, inplace=True)
      #verileri tarih sütununa göre sıralıyoruz.
      df.sort_values(by='observation_date', inplace=True)
    #Veri çerçevesini döndürüyoruz.
      return df