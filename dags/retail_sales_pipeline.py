
# Airflow DAG yapısını içe aktarıyoruz.
from airflow import DAG

# Python fonksiyonlarını Airflow görevi olarak çalıştırıyoruz.
from airflow.operators.python import PythonOperator

# DAG başlangıç tarihini belirlemek için kullanıyoruz.
from datetime import datetime

# Dosya yollarını yönetmek için kullanıyoruz.
from pathlib import Path

# Mevcut FRED veri çekme fonksiyonumuzu içe aktarıyoruz.
from src.data.ingestion import load_sales_data


# FRED verisini çekip CSV dosyasına kaydeden fonksiyon.
def extract_data():

    # Mevcut veri çekme fonksiyonumuzu çalıştırıyoruz.
    df = load_sales_data()

    # Verileri saklayacağımız klasörü belirliyoruz.
    output_dir = Path("/opt/airflow/data")

    # Klasör mevcut değilse oluşturuyoruz.
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV dosyasının tam yolunu oluşturuyoruz.
    output_path = output_dir / "sales_raw.csv"

    # DataFrame'i CSV dosyasına kaydediyoruz.
    df.to_csv(output_path, index=False)

    # İşlenen satır sayısını loglara yazdırıyoruz.
    print(f"Çekilen veri sayısı: {len(df)}")

    # Dosya yolunu döndürüyoruz.
    # PythonOperator bu değeri XCom'a kaydedebilir.
    return str(output_path)


# DAG yapılandırmamızı oluşturuyoruz.
with DAG(
    dag_id="retail_sales_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "retail-sales"],
) as dag:

    # İlk gerçek veri çekme görevimizi tanımlıyoruz.
    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )
