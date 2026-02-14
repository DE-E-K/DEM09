from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path so we can import src
PROJECT_ROOT = "/opt/airflow/dags/repo" # Adjust this based on actual Airflow volume mount
sys.path.append(PROJECT_ROOT)

default_args = {
    'owner': 'Flight Fare',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'flight_fare_retraining',
    default_args=default_args,
    description='A simple DAG to retrain the flight fare prediction model',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['machine_learning', 'flight_fare'],
) as dag:

    check_data_task = BashOperator(
        task_id='check_data_availability',
        bash_command=f'ls {PROJECT_ROOT}/data/raw/Flight_Price_Dataset_of_Bangladesh.csv',
    )

    # Assuming the environment where Airflow runs has all dependencies installed
    # Or use DockerOperator / KubernetesPodOperator for better isolation
    retrain_task = BashOperator(
        task_id='retrain_model',
        bash_command=f'cd {PROJECT_ROOT} && python -m src.main --run-all',
    )
    
    notify_task = BashOperator(
        task_id='notify_success',
        bash_command='echo "Model retraining completed successfully."',
    )

    check_data_task >> retrain_task >> notify_task
