import polars as pl
from .implicitBase import implicitBase
from ..utils.getRepositoryPath import getRepositoryPath

class Twitch100K(implicitBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/Twitch100K/100k_a.csv", has_header=False)
        df_RAW = df_RAW.filter( pl.col("column_5") - pl.col("column_4") >= 3 ) # 30分以上視聴
        df_RAW = df_RAW.select(
            pl.col("column_1").alias("user_id"),
            pl.col("column_3").alias("item_id"),
            pl.col("column_5").alias("timestamp")
        ).unique()
        
        # Sort
        df_RAW = df_RAW.sort("timestamp", "user_id", "item_id")

        # Set Variables
        self.df_RAW = df_RAW

        return self