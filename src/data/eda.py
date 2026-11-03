import pandas as pd
 #Grafik oluşturmak için matplotlib kütüphanesini içe aktarıyoruz.
import matplotlib.pyplot as plt

#Zaman serisi ayrıştırması için statsmodels kütüphanesini içe aktarıyoruz.
from statsmodels.tsa.seasonal import seasonal_decompose

#Satış verisinin temel istatistiklerini ve dağılımını analiz ediyoruz.
def analyze_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    
    #Satış sütununun temel istatistiklerini hesaplıyoruz.
    statiscs =df["sales"].describe()
    
    #Hesaplanan istatistikleri bir sözlük olarak döndürüyoruz.
    return statiscs.to_dict()

#Aylık satışların zaman içindeki  değişimini görselleştiriyoruz.
def plot_sales_trend(df: pd.DataFrame) -> None:
    
    #Grafik alanlarını oluşturuyoruz.
    plt.figure(figsize=(12, 6))
    
    #Tarihleri X, satışları Y eksenine yerleştiriyoruz.
    plt.plot(df.index, df["sales"], label="Monthly Sales", color="blue", marker="o")
    
    #Grafik başlığını ve eksen etiketlerini ekliyoruz.
    plt.title("US Retail Sales Over Time")
    
    #X ve Y eksenlerini etiketliyoruz
    plt.xlabel("Date")
    plt.ylabel("Sales (in millions of dollars)")
    
    #Açıklamaları ve ızgarayı ekliyoruz.
    plt.legend()
    
    #Grafik elemanlarının düzgün görünmesini sağlıyoruz.
    plt.tight_layout()
    
    #grafiği ekranda gösteriyoruz.
    plt.show()
    
#12 Aylık hareketli ortalama ile satış trendini görselleştiriyoruz.
def plot_rolling_mean(df: pd.DataFrame) -> None:

    # Son 12 ayın ortalamasını hesaplıyoruz.
    rolling_mean = df["sales"].rolling(window=12).mean()

    # Grafik alanını oluşturuyoruz.
    plt.figure(figsize=(12, 5))

    # Gerçek aylık satışları çiziyoruz.
    plt.plot(
        df.index,
        df["sales"],
        label="Monthly Sales",
        alpha=0.5
    )

    # Hareketli ortalamayı çiziyoruz.
    plt.plot(
        df.index,
        rolling_mean,
        label="12-Month Rolling Mean",
        linewidth=2
    )

    # Grafik başlığını belirliyoruz.
    plt.title("Retail Sales and 12-Month Rolling Mean")

    # Eksen adlarını ekliyoruz.
    plt.xlabel("Date")
    plt.ylabel("Sales (Million USD)")

    # Açıklama kutusunu gösteriyoruz.
    plt.legend()

    # Grafiğin yerleşimini düzenliyoruz.
    plt.tight_layout()

    # Grafiği gösteriyoruz.
    plt.show()

#Aylık mevsimsellik profilini hesaplıyoruz.

def analyze_monthly_seasonality(
    df: pd.DataFrame
) -> pd.Series:

    # Orijinal veriyi korumak için kopyalıyoruz.
    data = df.copy()

    # Tarih index'inden yıl bilgisini çıkarıyoruz.
    data["year"] = data.index.year

    # Tarih index'inden ay bilgisini çıkarıyoruz.
    data["month"] = data.index.month

    # Her yılın aylık ortalama satışını hesaplıyoruz.
    yearly_mean = data.groupby("year")["sales"].transform("mean")

    # Her ayın satışını kendi yılının ortalamasına bölüyoruz.
    data["seasonal_ratio"] = data["sales"] / yearly_mean

    # Yalnızca 12 ayı eksiksiz bulunan yılları seçiyoruz.
    complete_years = data.groupby("year")["sales"].transform("count") == 12
    data = data.loc[complete_years]

    # Her takvim ayı için ortalama mevsimsel oranı hesaplıyoruz.
    monthly_profile = data.groupby("month")["seasonal_ratio"].mean()

    # Aylık mevsimsellik profilini döndürüyoruz.
    return monthly_profile

def plot_seasonal_decomposition(df):
    #Satış sütununu seçiyoruz.
    sales_series = df["sales"]
    
    #Aylık veriyi 12 aylık mevsimsellik ile ayrıştırıyoruz.
    
    decomposition = seasonal_decompose(sales_series, 
                                       model="multiplicative",
                                       period=12)
    #Dört ayrı grafik alanı oluşturuyoruz.
    fig = decomposition.plot()
    
    #grafiğin boyutunu ayarlıyoruz.
    fig.set_size_inches(12, 9)
    
    #Grafik elemanlarını düzenliyoruz.
    plt.tight_layout()
    
    #Grafiği ekranda gösteriyoruz.
    plt.show()