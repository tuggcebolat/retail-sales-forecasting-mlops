# Airflow DAG yapısını içe aktarıyoruz.
from airflow import DAG

# Python fonksiyonlarını Airflow görevi olarak çalıştırıyoruz.
from airflow.operators.python import PythonOperator

# DAG başlangıç tarihini belirlemek için kullanıyoruz.
from datetime import datetime

# Dosya yollarını yönetmek için kullanıyoruz.
from pathlib import Path

# Veri işleme kütüphanemizi içe aktarıyoruz.
import pandas as pd

# Mevcut veri çekme fonksiyonumuzu içe aktarıyoruz.
from src.data.ingestion import load_sales_data

# Mevcut veri doğrulama fonksiyonumuzu içe aktarıyoruz.
from src.data.validation import validate_sales_data


# 1. VERİ ÇEKME
def extract_data():

    # FRED üzerinden satış verilerini çekiyoruz.
    df = load_sales_data()

    # Verilerin saklanacağı klasörü belirliyoruz.
    output_dir = Path("/opt/airflow/data")

    # Klasör mevcut değilse oluşturuyoruz.
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV dosyasının tam yolunu oluşturuyoruz.
    output_path = output_dir / "sales_raw.csv"

    # DataFrame'i CSV dosyasına kaydediyoruz.
    df.to_csv(output_path, index=False)

    # Çekilen veri sayısını logluyoruz.
    print(f"Çekilen veri sayısı: {len(df)}")

    # XCom Push: Dosya yolunu gönderiyoruz.
    return str(output_path)


# 2. VERİ DOĞRULAMA
def validate_data(ti):

    # XCom Pull: Önceki görevden dosya yolunu alıyoruz.
    raw_path = ti.xcom_pull(
        task_ids="extract_data",
        key="return_value"
    )

    # CSV dosyasını okuyoruz.
    df = pd.read_csv(
        raw_path,
        parse_dates=["observation_date"]
    )

    # Veri kalitesi kontrollerini gerçekleştiriyoruz.
    df = validate_sales_data(df)

    # Doğrulanan veri sayısını logluyoruz.
    print(f"Doğrulanan veri sayısı: {len(df)}")

    # Sonraki görev için dosya yolunu gönderiyoruz.
    return raw_path


# 3. DAG YAPILANDIRMASI
with DAG(
    dag_id="retail_sales_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "retail-sales"],
) as dag:

    # Veri çekme görevimizi tanımlıyoruz.
    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    # Veri doğrulama görevimizi tanımlıyoruz.
    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # Görevlerin çalışma sırasını belirliyoruz.
    extract_task >> validate_task