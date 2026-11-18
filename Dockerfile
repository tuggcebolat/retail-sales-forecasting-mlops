# Mevcut Airflow image'ını temel alıyoruz.
FROM apache/airflow:2.10.5-python3.11

# SARIMA için gerekli kütüphaneyi kuruyoruz.
RUN pip install --no-cache-dir statsmodels