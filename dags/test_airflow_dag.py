
# Airflow DAG sınıfını içe aktarıyoruz.
from airflow import DAG

# Python fonksiyonlarını Airflow görevi olarak çalıştıracağız.
from airflow.operators.python import PythonOperator

# DAG başlangıç tarihini tanımlamak için kullanıyoruz.
from datetime import datetime


# İlk test görevimizin çalıştıracağı fonksiyon.
def hello_airflow():
    print("Merhaba! Retail Sales Forecasting Airflow çalışıyor.")


# DAG'ımızın yapılandırmasını oluşturuyoruz.
with DAG(
    dag_id="retail_sales_test",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "retail-sales"],
) as dag:

    # Python fonksiyonumuzu Airflow görevine dönüştürüyoruz.
    test_task = PythonOperator(
        task_id="hello_airflow",
        python_callable=hello_airflow,
    )
