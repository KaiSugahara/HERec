import polars as pl
from .ratingBase import ratingBase
from ..utils.getRepositoryPath import getRepositoryPath

class Yelp(ratingBase):

    def read(self):

        # READ
        df_RAW = pl.read_ndjson( f"{getRepositoryPath()}/dataset/Yelp/yelp_training_set/yelp_training_set_review.json" )

        # Select Columns & Cast Datetime
        df_RAW = df_RAW.select(
            pl.col("user_id"),
            pl.col("business_id").alias("item_id"),
            pl.col("stars").alias("rating"),
            pl.col("date").str.to_date("%Y-%m-%d").alias("timestamp"),
        )

        # Sort
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self