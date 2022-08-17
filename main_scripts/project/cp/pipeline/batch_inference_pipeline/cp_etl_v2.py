import argparse
from datetime import date, timedelta

from rogers.cp.preprocessor import VerintETL
from pyspark.sql import DataFrame
from rogers.cp.dataOPs import Ium, AppDcc, Hem
from rogers.cp.preprocessor.ModemCallETL import ModemCallETL


def main(output_blob_folder):
    service_quality_df = Hem.load_data(Hem.modem_service_quality_table,
                                       condition="(channel_number =0)")
    accessibility_df = Hem.load_data(Hem.modem_accessibility_table)
    attainability_df = Hem.load_data(Hem.modem_attainability_table)
    intermittency_df = Hem.load_data(Hem.intermittency_daily_table)
    resi_df = Hem.load_data(Hem.resi_usage_ranked_daily_table)
    icm_cases_df = AppDcc.load_data(AppDcc.cbc_icm_cases_table)
    verint_df: DataFrame = VerintETL().etl()

    cp_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                          modem_attainability_df=attainability_df,
                          modem_service_quality_df=service_quality_df,
                          intermittency_df=intermittency_df,
                          icm_cases_df=icm_cases_df,
                          resi_df=resi_df,
                          agg_window=-20)
    df: DataFrame = cp_etl.etl()

    df = has_called(df, verint_df)
    inference_date = date.today() - timedelta(2)

    df = df.filter(df["event_date"] == inference_date)
    df = df.drop_duplicates(['account_number'])
    df = df.fillna(0)
    print('*** ETL COMPLETE ***')
    print("Number of output rows:", df.count())
    print('*** Writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')
    df.coalesce(1).write.mode("overwrite").option("header", True).parquet(output_blob_folder)
    print("Finished writing output")



def has_called(df: DataFrame, verint_df: DataFrame):
    verint_df = verint_df.withColumnRenamed("target", "called")

    df = df.join(verint_df, on=["account_number", "event_date"], how='left').fillna({'called': 0})

    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser("cp_etl_test")
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
