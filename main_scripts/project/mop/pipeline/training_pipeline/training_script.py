from azureml.core import Workspace, Experiment, Run, Datastore, Dataset
from azureml.interpret import ExplanationClient
from interpret_community.mimic import MimicExplainer
from interpret_community.mimic.models import LGBMExplainableModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, plot_roc_curve, plot_precision_recall_curve
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from datetime import date

today = date.today()

print(today)

def preprocessing(df):

    #Add modem manufacturer to this list
    final_df = df.drop(['mac', 'cmts_md_if_name', 'account', 'cmts',
                      'postal_zip_code', 'event_date', 'shub',
                      'us_port', 'smt', 'channel_number', 'model', 'modem_manufacturer', 'phub', 'is_port_lvl_flg',
                      'downtime_full_day', 'window'], axis=1)

    features = final_df.drop(['target'], axis=1)
    target = np.array(final_df['target'])

    return features, target


with mlflow.start_run():
    run = Run.get_context()
    client = ExplanationClient.from_run(run)
    ws = run.experiment.workspace
    datastore = ws.datastores.get("data_source_mazcacnpedigitaldls01_ml_etl_output_data")
    train_path = datastore.path(
        "MOP/training_data/part-00000-tid-7520578203294983843-b529b81f-a515-4a49-847e-3d3be89619fc-1305072-1.c000.snappy.parquet")
    test_path = datastore.path(
        "MOP/validation_data/part-00000-tid-8951171407885472901-a73922bd-1d3d-44af-b91d-a45242cb7dff-1475239-1.c000.snappy.parquet")
    train_df = Dataset.Tabular.from_parquet_files(path=train_path, validate=False)
    test_df = Dataset.Tabular.from_parquet_files(path=test_path, validate=False)
    train_df = train_df.to_pandas_dataframe()
    test_df = test_df.to_pandas_dataframe()

    X_train, y_train = preprocessing(train_df)
    X_test, y_test = preprocessing(test_df)

    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("ROC score: ", roc_auc)
    print("Precision score: ", precision)
    print("Recall score: ", recall)

    # log metrics
    mlflow.log_metric(f"ROC AUC", roc_auc)
    mlflow.log_metric(f"Recall", recall)
    mlflow.log_metric(f"Precision", precision)

    # plot and save roc auc curve
    # fig, ax = plt.subplots()
    # plot_roc_curve(model, X_test, y_test, drop_intermediate=False, ax=ax)
    # ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # ax.legend()
    # fig.savefig(f"RF_ROC_AUC_curve.png")
    # mlflow.log_artifact(f"RF_ROC_AUC_curve.png")

    # plot and save precision and recall curve
    # fig, ax = plt.subplots()
    # plot_precision_recall_curve(model, X_test, y_test, ax=ax)
    # ax.plot([0, 1], [0, 0], linestyle='--', label='No Skill')
    # ax.legend()
    # fig.savefig(f"{model} pr curve.png")
    # mlflow.log_artifact(f"{model} pr curve.png")

    # log the model
    mlflow.sklearn.log_model(model, f"mop_random_forest_v1_{today}")

    # model explanation
    explainer = MimicExplainer(model,
                               X_train,
                               LGBMExplainableModel,
                               augment_data=True,
                               max_num_of_augmentations=10,
                               features=X_train.columns,
                               )
    # explanation = explainer.explain_global(X_test)
    # df = explain_sklearn(model, X_train)
    # if df is not None:
    #     df.to_csv(f"{model}_explainability_{today}.csv", index=False)
    #     mlflow.log_artifact(f"{model}_explainability_{today}.csv")
    # explain model
    # original_model = run.register_model(model_name=f"mop_random_forest_v1_{today}",
    #                                     model_path=f"mop_random_forest_v1_{today}/model.pkl")
    # comment = f'Global explanation on {model} model trained on modem offline prediction dataset {today}'
    # client.upload_model_explanation(explanation, comment=comment, model_id=original_model.id)
