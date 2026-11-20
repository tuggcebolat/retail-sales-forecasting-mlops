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

# Veri ön işleme fonksiyonumuzu içe aktarıyoruz.
from src.data.preprocessing import preprocess_sales_data

#Özellik mühendisliği fonksiyonumuzu içe aktarıyoruz.
from src.features.feature_engineering import create_features

# SARIMA ve Seasonal Naive karşılaştırma fonksiyonunu içe aktarıyoruz.
from src.models.backtesting import compare_models_backtest

# Import the XGBoost backtesting function.
from src.models.xgboost_backtesting import xgboost_rolling_backtest

from src.models.model_selection import compare_and_select_models

from src.models.model_training import train_selected_model

from src.models.experiment_tracking import log_all_models
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

def preprocess_data(ti):

    # Önceki görevden dosya yolunu alıyoruz.
    raw_path = ti.xcom_pull(
        task_ids="validate_data",
        key="return_value"
    )

    # CSV dosyasını okuyup tarih sütununu dönüştürüyoruz.
    df = pd.read_csv(
        raw_path,
        parse_dates=["observation_date"]
    )

    # Veri ön işleme fonksiyonumuzu çalıştırıyoruz.
    df = preprocess_sales_data(df)

    # İşlenmiş verinin dosya yolunu belirliyoruz.
    output_path = "/opt/airflow/data/sales_processed.csv"

    # Tarih index'ini koruyarak veriyi kaydediyoruz.
    df.to_csv(output_path, index=True)

    # Dosya yolunu XCom ile gönderiyoruz.
    return output_path

def train_best_model(ti):

    # Retrieve the selected model's name.
    best_model = ti.xcom_pull(task_ids="compare_models")

    # Retrieve the preprocessed data path.
    processed_path = ti.xcom_pull(task_ids="preprocess_data")

    # Load historical sales data.
    df = pd.read_csv(
        processed_path,
        parse_dates=["observation_date"],
        index_col="observation_date"
    ).asfreq("MS")

    # Train the selected model.
    model = train_selected_model(df, best_model)

    print(f"Successfully trained: {best_model}")
    
def feature_engineering_data(ti):

    # XCom üzerinden işlenmiş verinin yolunu alıyoruz.
    processed_path = ti.xcom_pull(
        task_ids="preprocess_data",
        key="return_value"
    )

    # Tarih sütununu index yaparak CSV dosyasını okuyoruz.
    df = pd.read_csv(
        processed_path,
        parse_dates=["observation_date"],
        index_col="observation_date"
    )

    # Lag, rolling ve calendar özelliklerini oluşturuyoruz.
    df = create_features(df)

    # Yeni dosya yolunu belirliyoruz.
    output_path = "/opt/airflow/data/sales_features.csv"

    # Özellikleri oluşturulmuş veriyi kaydediyoruz.
    df.to_csv(output_path, index=True)

    # Dosya yolunu XCom ile gönderiyoruz.
    return output_path

def backtest_Statistical_models(ti):
    #retrieve the processed data path.
    processed_path = ti.xcom_pull(
        task_ids="preprocess_data",
        key="return_value"
    )
    
    #load the time  series data.
    df=pd.read_csv(
        processed_path,
        parse_dates=["observation_date"],
        index_col="observation_date"
        
    ).asfreq("MS")
    #Evaluate Seasonal Naive and SARIMA
    results =compare_models_backtest(df)
    
    #Save the evaluation results.
    output_path ="/opt/airflow/data/statistical_backtest_results.csv"
    results.to_csv(output_path, index=False)
    
    #return the output file path.
    return output_path

def backtest_xgboost(ti):

    # Retrieve the processed data path.
    processed_path = ti.xcom_pull(
        task_ids="preprocess_data",
        key="return_value"
    )

    # Load the time series data.
    df = pd.read_csv(
        processed_path,
        parse_dates=["observation_date"],
        index_col="observation_date"
    ).asfreq("MS")

    # Run XGBoost backtesting.
    results = xgboost_rolling_backtest(df)

    # Save the evaluation results.
    output_path = "/opt/airflow/data/xgboost_backtest_results.csv"
    results.to_csv(output_path, index=False)

    # Return the output file path.
    return output_path
    
def compare_models(ti):

    # Retrieve the statistical backtesting results path.
    statistical_path = ti.xcom_pull(
        task_ids="backtest_statistical_models"
    )

    # Retrieve the XGBoost backtesting results path.
    xgboost_path = ti.xcom_pull(
        task_ids="backtest_xgboost"
    )

    # Load and combine both datasets.
    df = pd.concat([
        pd.read_csv(statistical_path),
        pd.read_csv(xgboost_path)
    ], ignore_index=True)

    # Compare models and select the best one.
    summary, best_model = compare_and_select_models(df)
    log_all_models(summary)

    # Save the model comparison results.
    summary.to_csv(
        "/opt/airflow/data/model_comparison.csv",
        index=False
    )

    # Return the selected model's name.
    return best_model

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
    # Veri ön işleme görevimizi oluşturuyoruz.
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )
    # Özellik mühendisliği görevimizi tanımlıyoruz.
    feature_task = PythonOperator(
        task_id="feature_engineering_data",
        python_callable=feature_engineering_data,
    )
    #Define the backtesting task.
    backtest_task =PythonOperator(
        task_id ="backtest_statistical_models",
        python_callable =backtest_Statistical_models
    )
    # Define the XGBoost backtesting task.
    xgboost_task = PythonOperator(
        task_id="backtest_xgboost",
        python_callable=backtest_xgboost
    )
    # Define the model comparison task.
    comparison_task = PythonOperator(
       task_id="compare_models",
       python_callable=compare_models
    )
    # Define the automatic model training task.
    training_task = PythonOperator(
       task_id="train_best_model",
       python_callable=train_best_model
    )
     # Görevlerin çalışma sırasını belirliyoruz.
    extract_task >> validate_task >> preprocess_task
    preprocess_task >> [feature_task, backtest_task, xgboost_task]
    [backtest_task, xgboost_task] >> comparison_task >> training_task