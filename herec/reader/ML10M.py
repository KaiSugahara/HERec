import polars as pl
from .ratingBase import ratingBase

class ML10M(ratingBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv("../dataset/ML10M/ml-10M100K/ratings.dat", separator=":", has_header=False)
        df_RAW = df_RAW[:, 0::2]
        df_RAW.columns = ["user_id", "item_id", "rating", "timestamp"]
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self