import polars as pl
from .ratingBase import ratingBase
from ..utils.getRepositoryPath import getRepositoryPath

class ML25M(ratingBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/ML25M/ml-25m/ratings.csv")
        df_RAW.columns = ["user_id", "item_id", "rating", "timestamp"]
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self