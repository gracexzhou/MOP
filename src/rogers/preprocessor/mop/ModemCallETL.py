import logging
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *


class ModemCallETL:
    agg_dict = {"codeword_error_rate": ["avg","var", "kurtosis"], 'network_topology_key': ["avg","var", "kurtosis"],
                'us_util_total_mslots': ["avg","var", "kurtosis"],
                'us_util_ucastgranted_mslots': ["avg","var", "kurtosis"], 'us_util_total_cntn_mslots': ["avg","var", "kurtosis"],
                'us_util_used_cntn_mslots': ["avg","var", "kurtosis"], 'us_util_coll_cntn_mslots': ["avg","var", "kurtosis"],
                'us_util_total_cntn_req_mslots': ["avg","var", "kurtosis"],'us_util_used_cntn_req_mslots': ["avg", "var", "kurtosis"],
                'us_util_coll_cntn_req_mslots': ["avg","var", "kurtosis"],'us_util_total_cntn_req_data_mslots': ["avg", "var", "kurtosis"],
                'us_util_used_cntn_req_data_mslots': ["avg","var", "kurtosis"], 'us_util_coll_cntn_req_data_mslots': ["avg","var", "kurtosis"],
                'us_util_total_cntn_init_maint_mslots': ["avg","var", "kurtosis"],'us_util_used_cntn_init_maint_mslots': ["avg","var", "kurtosis"],
                'us_util_coll_cntn_init_maint_mslots': ["avg","var", "kurtosis"], 'us_bytes': ["avg","var", "kurtosis"],
                'capacity_bps': ["avg","var", "kurtosis"], 'us_throughput_bps': ["avg","var", "kurtosis"], 'us_throughput_max': ["avg","var", "kurtosis"],
                'us_throughput_bps_95': ["avg","var", "kurtosis"], 'us_throughput_bps_98': ["avg","var", "kurtosis"], 'rop': ["avg","var", "kurtosis"],
                'cmts_cm_us_rx_power_avg': ["avg","var", "kurtosis"], 'cmts_cm_us_signal_noise_avg': ["avg","var", "kurtosis"], 'cmts_cm_us_rx_power_sum': ["avg","var", "kurtosis"],
                'cmts_cm_us_signal_noise_sum': ["avg","var", "kurtosis"], 'cmts_cm_us_rx_power_max': ["avg","var", "kurtosis"], 'cmts_cm_us_signal_noise_max': ["avg","var", "kurtosis"],
                'cmts_cm_us_uncorrectables': ["avg","var", "kurtosis"], 'cmts_cm_us_correcteds': ["avg","var", "kurtosis"], 'cmts_cm_us_unerroreds': ["avg","var", "kurtosis"],
                '0100_hrs': ["avg", "var", "kurtosis"], '0600_hrs': ["avg", "var", "kurtosis"], '1200_hrs': ["avg", "var", "kurtosis"],
                '1800_hrs': ["avg", "var", "kurtosis"], '2400_hrs': ["avg", "var", "kurtosis"], 'outage_count':["avg", "var", "kurtosis"]
            }

    # agg_dict = {'codeword_error_rate': ["avg"],
    #             'us_util_total_mslots': ["avg", "var"],
    #             'us_util_ucastgranted_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_total_cntn_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_used_cntn_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_coll_cntn_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_total_cntn_req_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_used_cntn_req_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_used_cntn_req_data_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_total_cntn_init_maint_mslots': ["avg", "var", "kurtosis"],
    #             'us_util_coll_cntn_init_maint_mslots': ["avg", "var", "kurtosis"],
    #             'us_bytes': ["avg", "var", "kurtosis"],
    #             'us_throughput_bps': ["avg", "var", "kurtosis"],
    #             'us_throughput_max': ["avg", "var", "kurtosis"],
    #             'us_throughput_bps_95': ["avg","var"], 'us_throughput_bps_98': ["avg","var"],
    #             'cmts_cm_us_rx_power_avg': ["avg", "var", "kurtosis"],
    #             'cmts_cm_us_signal_noise_avg': ["avg", "var", "kurtosis"],
    #             'cmts_cm_us_rx_power_sum': ["avg", "var", "kurtosis"],
    #             'cmts_cm_us_signal_noise_sum': ["avg", "var", "kurtosis"],
    #             'cmts_cm_us_rx_power_max': ["avg", "var", "kurtosis"],
    #             'cmts_cm_us_signal_noise_max': ["avg", "var", "kurtosis"],
    #             'cmts_cm_us_uncorrectables': ["avg"],
    #             'cmts_cm_us_correcteds': ["avg", "var"],
    #             'cmts_cm_us_unerroreds': ["avg", "var"],
    #             'outage_count': ["avg", "var", "kurtosis"]
    #             }

    cols_to_scale = ['codeword_error_rate', 'network_topology_key', 'us_util_total_mslots',
                     'us_util_ucastgranted_mslots', 'us_util_total_cntn_mslots',
                     'us_util_used_cntn_mslots', 'us_util_coll_cntn_mslots', 'us_util_total_cntn_req_mslots',
                     'us_util_used_cntn_req_mslots', 'us_util_coll_cntn_req_mslots',
                     'us_util_total_cntn_req_data_mslots',
                     'us_util_used_cntn_req_data_mslots', 'us_util_coll_cntn_req_data_mslots',
                     'us_util_total_cntn_init_maint_mslots', 'us_util_used_cntn_init_maint_mslots',
                     'us_util_coll_cntn_init_maint_mslots', 'us_bytes',
                     'capacity_bps', 'us_throughput_bps', 'us_throughput_max', 'us_throughput_bps_95',
                     'us_throughput_bps_98', 'rop', 'cmts_cm_us_rx_power_avg', 'cmts_cm_us_signal_noise_avg',
                     'cmts_cm_us_rx_power_sum',
                     'cmts_cm_us_signal_noise_sum', 'cmts_cm_us_rx_power_max', 'cmts_cm_us_signal_noise_max',
                     'cmts_cm_us_uncorrectables', 'cmts_cm_us_correcteds', 'cmts_cm_us_unerroreds', '0100_hrs',
                     '0600_hrs', '1200_hrs',
                     '1800_hrs', '2400_hrs', 'outage_count']

    # cols_to_scale = ['codeword_error_rate', 'us_util_total_mslots',
    #                  'us_util_ucastgranted_mslots',
    #                  'us_util_total_cntn_mslots',
    #                  'us_util_used_cntn_mslots',
    #                  'us_util_coll_cntn_mslots',
    #                  'us_util_total_cntn_req_mslots',
    #                  'us_util_used_cntn_req_mslots',
    #                  'us_util_used_cntn_req_data_mslots',
    #                  'us_util_total_cntn_init_maint_mslots',
    #                  'us_util_coll_cntn_init_maint_mslots',
    #                  'us_bytes',
    #                  'capacity_bps', 'us_throughput_bps',
    #                  'us_throughput_max',
    #                  'us_throughput_bps_95', 'us_throughput_bps_98',
    #                  'cmts_cm_us_rx_power_avg',
    #                  'cmts_cm_us_signal_noise_avg',
    #                  'cmts_cm_us_rx_power_sum',
    #                  'cmts_cm_us_signal_noise_sum',
    #                  'cmts_cm_us_rx_power_max',
    #                  'cmts_cm_us_signal_noise_max',
    #                  'cmts_cm_us_uncorrectables',
    #                  'cmts_cm_us_correcteds',
    #                  'cmts_cm_us_unerroreds',
    #                  'outage_count']

    cols_to_keep_unscaled = ['event_date', 'mac', 'cmts', 'account', 'phub', 'shub', 'channel_number',
                             'cmts_md_if_name', 'smt', 'model', 'modem_manufacturer', 'postal_zip_code', 'us_port',
                             'is_port_lvl_flg', 'downtime_full_day']

    # cols_to_keep_unscaled = ['event_date', 'mac', 'cmts', 'account', 'phub', 'shub', 'channel_number',
    #                          'cmts_md_if_name', 'smt', 'model','postal_zip_code', 'us_port',
    #                          'is_port_lvl_flg', 'downtime_full_day']

    def __init__(self, modem_accessibility_df: DataFrame, modem_service_quality_df: DataFrame,
                 ccap_us_port_util_daily_df: DataFrame, agg_window: int):
        """
        :param modem_accessibility_df:
        :param modem_service_quality_df:
        :param modem_attainability_df:
        """

        self.logger = logging.getLogger(__name__)
        ccap_us_port_util_daily_df = ccap_us_port_util_daily_df.withColumnRenamed("cmts_host_name", "cmts")
        ccap_us_port_util_daily_df = ccap_us_port_util_daily_df.withColumnRenamed("date_key", "event_date")
        self.__df: DataFrame = modem_accessibility_df.join(modem_service_quality_df, ["mac", "account", "event_date"], "inner") \
            .join(ccap_us_port_util_daily_df, ["cmts", "cmts_md_if_index", "event_date"], "inner")
        # self.__df: DataFrame = anc_common_df

        self.__agg_window = Window.partitionBy("account").orderBy("event_date").rowsBetween(agg_window, 0)

    @staticmethod
    def extract(row):
        return (row.event_date, row.mac, row.cmts, row.account, row.phub, row.shub, row.channel_number,
                row.cmts_md_if_name, row.smt, row.model, row.modem_manufacturer, row.postal_zip_code, row.us_port,
                row.is_port_lvl_flg, row.downtime_full_day,) + tuple(row.scaledFeatures.toArray().tolist())

    def scaling_features(self, final_df):

        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        assembler = VectorAssembler().setInputCols(self.cols_to_scale).setOutputCol("features")
        sdf_transformed = assembler.setHandleInvalid("skip").transform(final_df)
        scaler_model = scaler.fit(sdf_transformed.select("features"))
        sdf_scaled = scaler_model.transform(sdf_transformed)

        final_df = sdf_scaled.select(*self.cols_to_keep_unscaled, "scaledFeatures").rdd \
            .map(self.extract).toDF(self.cols_to_keep_unscaled + self.cols_to_scale)

        return final_df

    def __apply_agg(self, column_name, agg):
        new_column = f"{column_name}_{agg}"

        def transform(df):
            if agg == "avg":
                return df.withColumn(new_column, f.avg(column_name).over(self.__agg_window))
            elif agg == "var":
                return df.withColumn(new_column, f.avg(f.pow(f.col(column_name), 2)).over(self.__agg_window) -
                                     f.pow(f.col(f"{column_name}_avg"), 2))
            elif agg == "sum":
                return df.withColumn(new_column, f.sum(column_name).over(self.__agg_window))
            elif agg == "max":
                return df.withColumn(new_column, f.max(column_name).over(self.__agg_window))
            elif agg == "kurtosis":
                return df.withColumn(new_column, f.avg(f.pow(f.col(column_name), 4)).over(self.__agg_window))

            else:
                return df

        return transform

    def etl(self):

        """
        ETL function to return a final pyspark dataframe for training dataset with targets appended
        :return:Dataframe
        """

        self.__df = self.scaling_features(self.__df)

        final_cols = ['mac', 'cmts', 'account', 'phub', 'shub', 'channel_number', 'cmts_md_if_name', 'smt', 'model',
                      'modem_manufacturer', 'postal_zip_code', 'us_port', 'is_port_lvl_flg', 'downtime_full_day', 'event_date', 'modem_offline']

        # final_cols = ['mac', 'cmts', 'account', 'phub', 'shub', 'channel_number', 'cmts_md_if_name', 'smt', 'model',
        #               'modem_manufacturer', 'postal_zip_code', 'us_port', 'is_port_lvl_flg', 'event_date',
        #               'target_output_value']

        for col in self.agg_dict:

            for agg_func in self.agg_dict[col]:
                new_column_name = f"{col}_{agg_func}"
                final_cols.append(new_column_name)

                self.__df = self.__df.transform(self.__apply_agg(col, agg_func))

        self.__df = self.__df.withColumn('modem_offline',
                           f.when((f.col("downtime_full_day") >= 180), 1).otherwise(0))

        return self.__df.select(*final_cols)
