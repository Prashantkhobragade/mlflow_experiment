# mlflow_experiment


# To run in remode server (Dagshub)
- connect git repo to Dagshub.
- export all the env variables of MLflow tracking (using git bash).
    - MLFLOW_TRACKING_URL
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWARD
- Add this code in app.py

      #for remote server only(DASGhub)
        remote_server_uri = MLFLOW_TRACKING_URL
        mlflow.set_tracking_uri(remote_server_uri)

- run python app.py (this won't create mlruns folder in local but will create it in a remote repo)

