import polars as pl
from .implicitBase import implicitBase
from herec.utils import *

class LFM2B_1MON(implicitBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/LFM2B_1MON/listening_events.tsv", separator="\t")
        df_RAW = df_RAW.with_columns( pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S") )
        df_RAW = df_RAW.filter( (pl.datetime(2020, 2, 20) <= pl.col("timestamp")) & (pl.col("timestamp") < pl.datetime(2020, 3, 20)) )
        df_RAW = df_RAW.select(
            pl.col("user_id"),
            pl.col("track_id").alias("item_id"),
            pl.col("timestamp"),
        ).unique()
        df_RAW = df_RAW.sort("timestamp", "user_id", "item_id")

        # Set Variables
        self.df_RAW = df_RAW

        return self