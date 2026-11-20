# Use the official Airflow image.
FROM apache/airflow:2.10.5-python3.11

# Install compatible dependencies.
RUN python -m pip install --no-cache-dir \
    apache-airflow==2.10.5 \
    statsmodels==0.15.0 \
    scikit-learn==1.9.1 \
    xgboost==3.2.0 \
    mlflow-skinny==2.19.0 \
    protobuf==4.25.6 \
    databricks-sdk==0.72.0 \
    cloudpickle==2.2.1 \
    joblib==1.5.3 \
    && python -m pip check