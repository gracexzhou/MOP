import argparse
import os
import pickle
import sys
from datetime import date
from azureml.core import Model, Run, Experiment, Workspace, Datastore, Dataset


def run_inference(input_data):
    # make inference
    input_data = input_data.fillna(0)
    # X = input_data[input_data.columns.difference(["account_number", "event_date", "join_date",
    #                                               "target", "window"])].values
    x = input_data.drop(['mac', 'cmts_md_if_name', 'account', 'cmts',
                         'postal_zip_code', 'event_date', 'shub',
                         'us_port', 'smt', 'channel_number', 'model', 'modem_manufacturer', 'phub', 'is_port_lvl_flg',
                         'downtime_full_day'], axis=1)

    pred = model.predict(x)
    score = model.predict_proba(x)[:, 1]

    # typecasting
    input_data['shub'] = input_data['shub'].astype("str")
    input_data['phub'] = input_data['phub'].astype("str")
    input_data['postal_zip_code'] = input_data['postal_zip_code'].astype("str")
    input_data['model'] = input_data['model'].astype("str")
    input_data['modem_manufacturer'] = input_data['modem_manufacturer'].astype("str")
    input_data["cmts_md_if_name"] = input_data['cmts_md_if_name'].astype("str")

    # cleanup
    input_data['pred'] = pred
    input_data["score"] = score

    return input_data[["mac", "shub", "model", "modem_manufacturer",
                       "cmts_md_if_name", "postal_zip_code", "phub", "account", "event_date", "pred", "score"]]


def run_saving():
    # Save the split data
    print("Saving Data...")
    print(inference_pd)
    batch_date = date.today()
    to_date_str = batch_date.strftime('%Y-%m-%d')
    # *** Write to LATEST folder ***
    output_path = os.path.join(output_dir, 'LATEST')
    print("Latest Path: ", output_path)
    os.makedirs(output_path, exist_ok=True)
    inference_pd.to_parquet(os.path.join(output_path, 'mop_prediction.parquet'), index=False)

    # *** Write to ARCHIVE folder ***
    output_path = os.path.join(output_dir, 'ARCHIVE', f"DATE_PART={to_date_str}")
    print("Archive Path: ", output_path)
    os.makedirs(output_path, exist_ok=True)
    inference_pd.to_parquet(os.path.join(output_path, 'mop_prediction.parquet'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--model_name', dest="model_name", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    model_name = args.model_name
    print(model_name)
    print(output_dir)

    # Get the experiment run context
    run = Run.get_context()
    ws = run.experiment.workspace

    model_path = Model.get_model_path(args.model_name)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    etl_df = run.input_datasets["etl_output"].to_pandas_dataframe()
    if etl_df.empty:
        print("Empty Dataframe, missing data")
        run.complete()
        sys.exit()
    else:
        etl_df["account"] = etl_df["account"].astype(str)
        inference_pd = run_inference(etl_df)
        run_saving()
