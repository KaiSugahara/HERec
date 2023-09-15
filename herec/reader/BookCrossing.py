import polars as pl
from .ratingBase import ratingBase
from herec.utils import *

class BookCrossing(ratingBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/BookCrossing/BX-Book-Ratings.csv", separator=";", encoding='latin-1')
        df_RAW.columns = ["user_id", "item_id", "rating"]

        # Set Variables
        self.df_RAW = df_RAW

        return self