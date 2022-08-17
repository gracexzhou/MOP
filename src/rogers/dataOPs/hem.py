from rogers.SparkOPs.databricks import DataBricks
import logging
from rogers.dataOPs.table_extract import TableExtract
logger = logging.getLogger("hem_DataOPs")

class Hem(TableExtract):
    """
        This class wraps data operations for hem database
    """

    ### Table names & queries on Databricks ###
    modem_accessibility_table = "delta_hem.modem_accessibility_daily"
    modem_attainability_table = "delta_hem.modem_attainability_daily"
    modem_service_quality_table = "delta_hem.modem_service_quality_daily"
    intermittency_daily_table = "delta_hem.intermittency_daily"
    resi_usage_ranked_daily_table = "delta_hem.resi_usage_ranked_daily"
    ccap_us_port_util_daily_table = "delta_hem.ccap_us_port_util_daily_fct"
    mac_usage_fct_table = "delta_hem.mac_usage_fct"

    all_tables = [modem_accessibility_table, modem_attainability_table, modem_service_quality_table,
                  intermittency_daily_table, resi_usage_ranked_daily_table, mac_usage_fct_table]
    columns = {modem_service_quality_table: "mac,cmts,account, phub, shub,channel_number,cmts_md_if_index,"
                                            "cmts_md_if_name,smt,modem_manufacturer,model,address,postal_zip_code,"
                                            "0100_hrs,0600_hrs,1200_hrs, 1800_hrs,2400_hrs,codeword_error_rate,"
                                            "cmts_cm_us_rx_power_sum,cmts_cm_us_signal_noise_sum,"
                                            "cmts_cm_us_rx_power_max,cmts_cm_us_signal_noise_max,"
                                            "cmts_cm_us_rx_power_avg,cmts_cm_us_signal_noise_avg,"
                                            "cmts_cm_us_uncorrectables,cmts_cm_us_correcteds,cmts_cm_us_unerroreds,"
                                            "event_date",

               modem_attainability_table: "CM_MAC_ADDR as mac, US_SPEED_ATTAINABLE_FULL , "
                                          "DS_SPEED_ATTAINABLE_FULL, Attainability_Pct , event_date,"
                                          "UP_MBYTES/1000 as up_gb, DOWN_MBYTES/1000 as down_gb",

               modem_accessibility_table: "mac, account, outage_count, "
                                          "ACCESSIBILITY_PERC_FULL_DAY/100 as ACCESSIBILITY_PERC_FULL_DAY, "
                                          "ACCESSIBILITY_PERC_PRIME/100 as ACCESSIBILITY_PERC_PRIME, downtime_full_day,"
                                          "event_date",

               intermittency_daily_table: "cm_mac_addr as mac, Intermittent_hrs, transitions, Offline_counts,"
                                          "event_date",

               resi_usage_ranked_daily_table: "cm_mac_addr as mac, event_date, hsi_down_1d_rank, hsi_up_1d_rank,"
                                              "IPTV_Down_1D_rank, IPTV_Up_1D_rank, "
                                              "RHP_Down_1D_rank, RHP_Up_1D_rank",

               ccap_us_port_util_daily_table: "network_topology_key, cmts_host_name, cmts_md_if_index, us_port, "
                                              "interface_count, is_port_lvl_flg, us_util_total_mslots,"
                                              "us_util_ucastgranted_mslots, us_util_total_cntn_mslots,"
                                              "us_util_used_cntn_mslots, us_util_coll_cntn_mslots,"
                                              "us_util_total_cntn_req_mslots, us_util_used_cntn_req_mslots,"
                                              "us_util_coll_cntn_req_mslots, us_util_total_cntn_req_data_mslots, "
                                              "us_util_used_cntn_req_data_mslots, us_util_coll_cntn_req_data_mslots,"
                                              "us_util_total_cntn_init_maint_mslots, "
                                              "us_util_used_cntn_init_maint_mslots,"
                                              "us_util_coll_cntn_init_maint_mslots,utilization_max,utilization_95,"
                                              "utilization_98,us_bytes,capacity_bps,us_throughput_bps,"
                                              "us_throughput_max,us_throughput_bps_95,us_throughput_bps_98,rop, "
                                              "date_key",
               }

    @staticmethod
    def load_data(table, condition=None):
        """
        Loading ela_rcis tables data to spark dataframes

        :return: spark dataframe
        """

        query = f"select {Hem.columns[table]} from {table}"
        if condition is not None:
            query = f"{query} where {condition}"

        logger.info(f"reading {table} from Databricks.")
        logger.debug(f"running query: {query}\n")

        return DataBricks.spark.sql(query)

    @staticmethod
    def load_all_data():
        """
        Loading all tables from hem into pyspark dataframes
        :return: list of pyspark dataframes
        """
        queries = {table: f"select {Hem.columns[table]} from {table}" for table in Hem.all_tables}
        logger.info(f"reading all hem tables from Databricks.")
        logger.debug(f"running queries: {[queries[table] for table in queries]}\n")

        return {table: DataBricks.spark.sql(queries[table]) for table in queries}
