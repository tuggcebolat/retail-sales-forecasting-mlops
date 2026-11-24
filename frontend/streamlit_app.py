# Kullanıcı arayüzü oluşturmak için.
import streamlit as st
import requests
import pandas as pd
# Sayfanın genel ayarlarını belirliyoruz.
st.set_page_config(
    page_title="Retail Sales Forecasting",
    layout="wide"
)

# Sayfanın ana başlığını oluşturuyoruz.
st.title("Retail Sales Forecasting")

# Uygulamanın amacını açıklıyoruz.
st.write("SARIMA Based Monthly Sales Forecasting")

# Kullanıcının tahmin süresini seçmesini sağlıyoruz.
steps = st.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=12,
    value=3
)

# Kullanıcının seçtiği ay sayısını gösteriyoruz.
st.write("Selected forecast period:", steps)

# FastAPI tahmin endpoint'inin adresi.
API_URL = "http://127.0.0.1:8000/predict"

if st.button("Generate Forecast"):

    try:
        # FastAPI'ye tahmin isteği gönderiyoruz.
        response = requests.post(
            API_URL,
            json={"steps": steps},
            timeout=30
        )

        # HTTP yanıtının başarılı olup olmadığını kontrol ediyoruz.
        response.raise_for_status()

        # JSON yanıtını Python sözlüğüne dönüştürüyoruz.
        result = response.json()

    except requests.exceptions.RequestException:
        # Bağlantı veya HTTP hatasında kullanıcıyı bilgilendiriyoruz.
        st.error("Forecast service is unavailable. Please try again.")

        # Kodun devam etmesini engelliyoruz.
        st.stop()

    # Tahmin listesini alıyoruz.
    forecast_data = result["forecasts"]

    # DataFrame oluşturuyoruz.
    df = pd.DataFrame(forecast_data)

    # Tablo başlığını oluşturuyoruz.
    st.subheader("Monthly Sales Forecast")

    # Tahmin tablosunu gösteriyoruz.
    st.dataframe(df, hide_index=True)
    # Tarih sütununu datetime veri tipine dönüştürüyoruz.
    df["date"] = pd.to_datetime(df["date"])

    # Grafik başlığını oluşturuyoruz.
    st.subheader("Sales Forecast Trend")

    # Satış tahminlerini çizgi grafiğinde gösteriyoruz.
    st.line_chart(
        df,
        x="date",
        y="predicted_sales"
    )
    # Ekranı üç sütuna ayırıyoruz.
    col1, col2, col3 = st.columns(3)

    # Toplam tahmini satış.
    col1.metric(
        "Total Forecast",
        f"{df['predicted_sales'].sum():,.2f}"
    )

    # Ortalama aylık satış.
    col2.metric(
        "Average Monthly Sales",
        f"{df['predicted_sales'].mean():,.2f}"
    )

    # En yüksek aylık satış tahmini.
    col3.metric(
        "Peak Monthly Sales",
        f"{df['predicted_sales'].max():,.2f}"
    )
    # DataFrame'i CSV formatına dönüştürüyoruz.
    csv_data = df.to_csv(index=False).encode("utf-8")

    # Kullanıcının CSV dosyasını indirmesini sağlıyoruz.
    st.download_button(
        label="Download Forecast CSV",
        data=csv_data,
        file_name="sales_forecast.csv",
        mime="text/csv",
        on_click="ignore"
    )