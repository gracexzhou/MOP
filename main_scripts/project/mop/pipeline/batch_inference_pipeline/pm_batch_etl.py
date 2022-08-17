import argparse
from datetime import date, timedelta
from pyspark.sql import DataFrame
from rogers.dataOPs.hem import Hem
from rogers.preprocessor.mop.ModemCallETL import ModemCallETL


def main(output_blob_folder):
    print(output_blob_folder)
    today = date.today()
    previous = today - timedelta(9)
    x = "event_date >= '{}'".format(previous)
    y = "date_key >= '{}'".format(previous)
    service_quality_df = Hem.load_data(Hem.modem_service_quality_table, condition=x)
    accessibility_df = Hem.load_data(Hem.modem_accessibility_table, condition=x)
    ccap_port_daily_df = Hem.load_data(Hem.ccap_us_port_util_daily_table, condition=y)

    pm_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                          ccap_us_port_util_daily_df=ccap_port_daily_df,
                          modem_service_quality_df=service_quality_df,
                          agg_window=-7)

    df: DataFrame = pm_etl.etl()
    inference_date = today - timedelta(1)
    # inference_date = today
    print("Inference_date: ", inference_date)
    df = df.filter(df["event_date"] == inference_date)
    print("Dataframe count on {0}: {1}".format(inference_date, df.count()))
    #df = df.limit(10000)
    df = df.drop_duplicates(['account'])
    #df = df.fillna(0)
    print('*** ETL COMPLETE ***')
    print("Number of output rows:", df.count())
    print('*** Writing FINAL MODEM OFFLINE PREDICTION OUTPUT To Azure Storage Blob ***')
    df.coalesce(1).write.mode("overwrite").option("header", True).parquet(output_blob_folder)
    print("Finished writing output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mop_etl_test")
    parser.add_argument("--etl_output")
    parser.add_argument("--AZUREML_SCRIPT_DIRECTORY_NAME")
    parser.add_argument("--AZUREML_RUN_TOKEN")
    parser.add_argument("--AZUREML_RUN_TOKEN_EXPIRY")
    parser.add_argument("--AZUREML_RUN_ID")
    parser.add_argument("--AZUREML_ARM_SUBSCRIPTION")
    parser.add_argument("--AZUREML_ARM_RESOURCEGROUP")
    parser.add_argument("--AZUREML_ARM_WORKSPACE_NAME")
    parser.add_argument("--AZUREML_ARM_PROJECT_NAME")
    parser.add_argument("--AZUREML_SERVICE_ENDPOINT")
    parser.add_argument("--AZUREML_EXPERIMENT_ID")
    parser.add_argument("--AZUREML_WORKSPACE_ID")
    parser.add_argument("--MLFLOW_EXPERIMENT_ID")
    parser.add_argument("--MLFLOW_EXPERIMENT_NAME")
    parser.add_argument("--MLFLOW_RUN_ID")
    parser.add_argument("--MLFLOW_TRACKING_URI")
    parser.add_argument("--MLFLOW_TRACKING_TOKEN")
    args = parser.parse_args()
    output_blob_folder = args.etl_output
    main(output_blob_folder)

