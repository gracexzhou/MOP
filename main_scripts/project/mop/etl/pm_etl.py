import sys
from rogers.mop.preprocessor.ModemCallETL import ModemCallETL
from rogers.mop.dataOPs.hem import Hem
from rogers.mop.sparkOPs.databricks import DataBricks
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as f
from pyspark.sql.functions import col



def main():

    service_quality_df = Hem.load_data(Hem.modem_service_quality_table, condition="event_date >= '2022-04-01' and event_date <= '2022-04-30'")
    accessibility_df = Hem.load_data(Hem.modem_accessibility_table, condition="event_date >= '2022-04-01' and event_date <= '2022-04-30'")
    ccap_port_daily_df = Hem.load_data(Hem.ccap_us_port_util_daily_table, condition="date_key >= '2022-04-01' and date_key <= '2022-04-30'")
    # anc_common_etl_df = Hem.load_data(Hem.anc_common_table, condition="event_date >= '2022-04-01' and event_date <= '2022-04-07'")
    windows = [1]
    for window in windows:
        pm_etl = ModemCallETL(modem_accessibility_df=accessibility_df,
                              ccap_us_port_util_daily_df=ccap_port_daily_df,
                              modem_service_quality_df=service_quality_df,
                              agg_window=-7)

        df: DataFrame = pm_etl.etl()
        df = window_target(df, window)
        train_df = df.where("event_date BETWEEN '2022-04-01' AND '2022-04-10'")
        # test_df = df.where("event_date BETWEEN '2022-04-08' AND '2022-04-15'")
        print("Train Columns: ", len(train_df.columns))
        # print("Test Columns: ", len(test_df.columns))
        major_df = train_df.filter(col('target') == 0)
        minor_df = train_df.filter(col('target') == 1)
        ratio = 46
        print("ratio: {}".format(ratio))
        sampled_majority_df = major_df.sample(False, 1 / ratio)
        us_train_df = sampled_majority_df.unionAll(minor_df)
        # us_train_df = us_train_df.limit(10000)
        sas_key = "BGU/Sg/TS40/bywqYA88MxU+cRV45uGZhj4X7v0iLPtQscClTQ6FgczU+0crsMj8gnDskj0NYTaitifNZiNDkA=="  # parser.SAS_KEY
        storage_name = "mazcacnpedigitaldls01"  # parser.STORAGE_NAME
        container_name = "ml-etl-output-data"  # parser.CONTAINER_NAME
        db = DataBricks(container_name, storage_name, sas_key)
        db.write_to_storage(us_train_df, f'modem_offline_prediction/training')
        print('*** Completed writing FINAL MOP TRAINING BASE To Azure Storage Blob ***')
        # db.write_to_storage(test_df, f'modem_offline_prediction/test')
        # print('*** Completed writing FINAL MOP TEST BASE To Azure Storage Blob ***')


# def undersampling(df: DataFrame):
#     major_df = df.filter(col('TARGET_OUTPUT_VALUE') == 0)
#     minor_df = df.filter(col('TARGET_OUTPUT_VALUE') == 1)
#     ratio = 24
#     print("ratio: {}".format(ratio))
#     sampled_majority_df = major_df.sample(False, 1 / ratio)
#     combined_df = sampled_majority_df.unionAll(minor_df)
#     return combined_df

def window_target(df: DataFrame, window: int = 10):
    """
    :param df:
    :param window:
    :return:
    """
    days = lambda i: i * 86400
    modem_offline_count_w = Window().partitionBy("account") \
        .orderBy(f.col("event_date")
                 .cast("timestamp")
                 .cast("long")) \
        .rangeBetween(days(1), days(window))
    # df = df.withColumn("window", f.lit(window))
    df = df.withColumn(f"target", f.sum("modem_offline").over(modem_offline_count_w))
    df = df.withColumn(f"target", f.when(f.col(f"target") > 0, 1).otherwise(0))
    return df

if __name__ == "__main__":
    main()
