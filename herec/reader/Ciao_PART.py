import polars as pl
from .ratingBase import ratingBase
from herec.utils import *

class Ciao_PART(ratingBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/Ciao_PART/ciao_with_rating_timestamp_txt/rating_with_timestamp.txt", separator=" ", has_header=False)
        df_RAW = df_RAW.select(
            pl.col("column_3").alias("user_id").cast(int),
            pl.col("column_5").alias("item_id").cast(int),
            pl.col("column_7").alias("category_id").cast(int),
            pl.col("column_9").alias("rating").cast(int),
            pl.col("column_11").alias("helpfulness").cast(int),
            pl.col("column_13").alias("timestamp").cast(int),
        )
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self