import mlflow

# Set the tracking URI to the current directory
mlflow.set_tracking_uri("mlruns/")

# Set the experiment name
mlflow.set_experiment("TestExperiment")

with mlflow.start_run():
    # Log a parameter
    mlflow.log_param("param1", "value1")

    # Log a metric
    mlflow.log_metric("metric1", 1.23)