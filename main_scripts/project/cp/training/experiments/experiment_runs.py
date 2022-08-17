from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, ComputeTarget, Dataset, Datastore

def main():
    print("starting experiment")
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='cp_ium_features')
    aml_compute = ComputeTarget(workspace=ws, name="npedigitalamlc01")
    # blob_datastore_name = "data_source_mazcacnpedigitaldls01_ml_etl_output_data"
    # ds = Datastore.get(ws, blob_datastore_name)
    # train_dataset_path = f'call_prediction/cp_train_ium_15/*.parquet'
    # val_dataset_path = f'call_prediction/cp_test_ium_15/*.parquet'
    # train_df = Dataset.Tabular.from_parquet_files(path=(ds, train_dataset_path))
    # test_df = Dataset.Tabular.from_parquet_files(path=(ds, val_dataset_path))
    # print("found datasets")
    config = ScriptRunConfig(source_directory='main_scripts/training/experiments',
                             script='training_script.py',
                             compute_target=aml_compute)
    env = Environment.get(ws,"cp_training_env")
    config.run_config.environment = env
    print("submitting experiment")
    run = experiment.submit(config)

if __name__=="__main__":
    main()