from azureml.core import Run, Dataset
from azureml.interpret import ExplanationClient
from interpret_community.mimic import MimicExplainer
from interpret_community.mimic.models import LGBMExplainableModel

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, \
    VotingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, plot_roc_curve, plot_precision_recall_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date

today = date.today()


def explain_sklearn(model, features):
    features_score = {}
    try:
        scores = model.feature_importances_
    except AttributeError:
        return None
    for idx, feature in enumerate(features):
        features_score[feature] = scores[idx]

    return pd.DataFrame(features_score.items(), columns=["feature", "importance"])


with mlflow.start_run():
    run = Run.get_context()
    # train_df = run.input_datasets['train_set'].to_pandas_dataframe()
    # test_df = run.input_datasets["test_set"].to_pandas_dataframe()
    client = ExplanationClient.from_run(run)
    ws = run.experiment.workspace

    shapes = [(1000,10), (1000,30), (1000,50), (1000,70), (1000,90), (600000,10), (600000,30), (600000,50),(600000,80),
              (600000,100)]

    for shape in shapes:

        X_train = pd.DataFrame(np.random.random(shape),columns=[f"col_{x}" for x in range(shape[1])])
        y_train = np.random.binomial(1,0.5,size=shape[0])
        X_test = pd.DataFrame(np.random.random(shape),columns=[f"col_{x}" for x in range(shape[1])])
        y_test = np.random.binomial(1,0.5,size=shape[0])

        mlflow.log_text(f"Training shape: {X_train.shape}\nTest shape:{X_test.shape}\n"
                        f"Columns: {X_train.columns}", "Data metrics.txt")

        model = GradientBoostingClassifier()

        model_name = f"random_forest_{shape[0]}_{shape[1]}"


        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)


        # log the model
        mlflow.sklearn.log_model(model, f"amlexplain_{model_name}_{today}")

        # model explanation
        explainer = MimicExplainer(model,
                                   X_train,
                                   LGBMExplainableModel,
                                   augment_data=True,
                                   max_num_of_augmentations=10,
                                   features=list(X_train.columns.values))
        explanation = explainer.explain_global(X_test)

        # explain model
        original_model = run.register_model(model_name=f"cp_{model_name}_{today}",
                                            model_path=f"cp_{model_name}_{today}/model.pkl")
        comment = f'Global explanation on {model_name} model trained on call prediction dataset {today}'
        client.upload_model_explanation(explanation, comment=comment, model_id=original_model.id)
