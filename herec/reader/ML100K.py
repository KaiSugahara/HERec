import polars as pl
from .ratingBase import ratingBase

class ML100K(ratingBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv("../dataset/ML100K/ml-100k/u.data", separator="\t", new_columns=["user_id", "item_id", "rating", "timestamp"], has_header=False)
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self