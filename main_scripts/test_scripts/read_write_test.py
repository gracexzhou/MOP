from rogers.dataOPs import Hem
from rogers.SparkOPs import DataBricks

"""
    Test: Reading dataOPs from Databricks and writing a sample to blob storage
"""

df = Hem.load_data(Hem.modem_accessibility_table)
storage_name = "mazcacnpedigitaldls01"#parser.STORAGE_NAME
container_name = "ml-etl-output-data"#parser.CONTAINER_NAME
db = DataBricks(container_name,storage_name)
db.write_to_storage(df.limit(100),"sample_100_accessibility")


