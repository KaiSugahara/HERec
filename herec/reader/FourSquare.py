import polars as pl
from .implicitBase import implicitBase
from ..utils.getRepositoryPath import getRepositoryPath

class FourSquare(implicitBase):

    def read(self):

        # READ
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/FourSquare/dataset_TSMC2014_TKY.csv")
        df_RAW = df_RAW.select(
            pl.col("userId").alias("user_id"),
            pl.col("venueId").alias("item_id"),
            pl.col("utcTimestamp").str.to_datetime("%a %b %d %T %z %Y").alias("timestamp"),
        )
        df_RAW = df_RAW.unique()

        # Sort
        df_RAW = df_RAW.sort("timestamp", "user_id", "item_id")

        # Set Variables
        self.df_RAW = df_RAW

        return self