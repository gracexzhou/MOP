import logging

from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql import Window

from rogers.dataOPs.verint import Verint


class VerintETL:

    def __init__(self):
        """
        Constructor for Verint ETL
        """

        self.logger = logging.getLogger(__name__)
        self.sessions_categories_df: DataFrame = Verint.load_data(Verint.sessions_categories_table)
        self.sessions_booked_df: DataFrame = Verint.load_data(Verint.sessions_booked_table)
        self.categories_df: DataFrame = Verint.load_data(Verint.categories_table)
        self.cbu_rog_conv_sumfct_df: DataFrame = Verint.load_data(Verint.cbu_rog_conv_sumfct_table)

    def etl(self):
        """
        ETL function to return a final pyspark dataframe for training dataset
        :return:Dataframe
        """
        categories_filter = [
            "Cable Customer Frustration",
            "L1R - Tech Issues: Internet",
            "BP1 Technical Support",
            "BP3 Technical Support",
            "HOT TOPIC: National Outage",
            "Ignite TV",
            "L1R - Tech Issues: TV",
            "L2R - INT: Modem Offline",
            "HOT TOPIC: Outage Compensation",
            "L1F - Tech Issues: Internet",
            "L2R - General Inquiries: Internet"
        ]
        cat_name_df = self.sessions_categories_df.join(self.categories_df, on=["category_key"], how="left"). \
            filter(f.col("category_name").isin(categories_filter))
        # cat_name_df = cat_name_df.select("sid", "category_name").distinct()
        self.cbu_rog_conv_sumfct_df = self.cbu_rog_conv_sumfct_df.where("event_date> '2021-01-01'")
        inter_df = self.cbu_rog_conv_sumfct_df.join(self.sessions_booked_df, on=["speech_id_verint"])
        final_df: DataFrame = inter_df.join(cat_name_df, on=["sid"])
        final_df = final_df.withColumn("target", f.lit(1))
        return final_df.select("account_number", "event_date", "target").distinct()
