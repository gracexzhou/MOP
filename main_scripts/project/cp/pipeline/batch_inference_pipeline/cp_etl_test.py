import argparse
from datetime import date, timedelta

from rogers.cp.preprocessor import VerintETL
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from rogers.cp.dataOPs import Ium, AppDcc, Hem
from rogers.cp.preprocessor.ModemCallETL import ModemCallETL


def main(output_blob_folder):
    service_quality_df = Hem.load_data(Hem.modem_service_quality_table,
                                       condition="(channel_number =0) and (event_date BETWEEN '2021-09-30' and '2021-10-10')")
    accessibility_df = Hem.load_data(Hem.modem_accessibility_table, condition="event_date BETWEEN '2021-09-30' and '2021-10-10'")
    attainability_df = Hem.load_data(Hem.modem_attainability_table, condition="event_date BETWEEN '2021-09-30' and '2021-10-10'")
    intermittency_df = Hem.load_data(Hem.intermittency_daily_table, condition="event_date BETWEEN '2021-09-30' and '2021-10-10'")
    ium_df = Ium.load_data(Ium.resi_usage_ranked_daily_table, condition="event_date BETWEEN '2021-09-30' and '2021-10-10'")
    icm_cases_df = AppDcc.load_data(AppDcc.cbc_icm_cases_table)


    cp_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                          modem_attainability_df=attainability_df,
                          modem_service_quality_df=service_quality_df,
                          intermittency_df=intermittency_df,
                          icm_cases_df=icm_cases_df,
                          resi_df=ium_df,
                          agg_window=-20)
    df: DataFrame = cp_etl.etl()

    # df = has_called(df, verint_df)
    df = df.withColumn("called", f.lit(0))
    print("Number of output rows:", df.count())
    df = df.fillna(0)
    df = df.sample(fraction=0.001).limit(100)

    print('*** ETL COMPLETE ***')
    print("Number of output rows:", df.count())
    print('*** Writing FINAL CALL PREDICTION OUTPUT To Azure Storage Blob ***')
    df.coalesce(1).write.mode("overwrite").option("header", True).parquet(output_blob_folder)
    print("Finished writing output")


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
