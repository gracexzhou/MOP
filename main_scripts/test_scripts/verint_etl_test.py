
from pyspark.sql import DataFrame

from rogers.preprocessor.cp.VerintETL import VerintETL


def main():
    verint_etl:DataFrame = VerintETL().etl()
    print("count is ", verint_etl.sample(fraction=0.1).count(), "end")
    print(verint_etl.columns)






if __name__ == "__main__":
    main()
