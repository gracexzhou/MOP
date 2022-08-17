import logging
from azureml.core.dataset import Dataset
from com.rogers.mlops.aml.training import AutoMlClassification
from com.rogers.mlops.aml.workspace import AMLObject
from azureml.train.automl import AutoMLConfig



def main():
    ws = AMLObject().ws
    val_data = Dataset.get_by_name(workspace=ws, name="call_prediction_val1")
    test_data = Dataset.get_by_name(workspace=ws, name="call_prediction_test1")
    config = {
        "primary_metric": 'AUC_weighted',
        "enable_early_stopping": True,
        "experiment_timeout_minutes": 480,
        "iterations": 5,
        "allowed_models": ["XGBoostClassifier", "RandomForest"],
        "verbosity": logging.INFO,
        "validation_data": val_data,
        "test_data": test_data,
        "featurization": 'off'

    }
    call_prediction_cls = AutoMlClassification(dsc="call_prediction_model1", ws=ws,
                                               training_data_name="call_prediction_train1",
                                               label_col_name='target',
                                               compute_target="npedigitalamlc01",
                                               config=config)
    best_run, fitted_model = call_prediction_cls.execute(experiment_name="call_prediction_experiment1")
    print(fitted_model)
    print(best_run)


if __name__ == '__main__':
    main()
