import polars as pl
from .implicitBase import implicitBase
from ..utils.getRepositoryPath import getRepositoryPath

class LastFM_TAG(implicitBase):

    def read(self):

        # READ
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/LastFM_TAG/user_taggedartists-timestamps.dat", separator="\t")

        # Select Columns
        df_RAW = df_RAW.select(
            pl.col("userID").alias("user_id"),
            pl.col("artistID").alias("item_id"),
            pl.col("timestamp"),
        )

        # Sort
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self