import polars as pl
from .implicitBase import implicitBase
from herec.utils import *

class ML100K_IMPLICIT(implicitBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/ML100K/ml-100k/u.data", separator="\t", new_columns=["user_id", "item_id", "rating", "timestamp"], has_header=False)
        df_RAW = df_RAW.sort("timestamp")
        
        # Keep only highly rated interactions
        df_RAW = df_RAW.filter( pl.col("rating") > 3.5 )

        # Set Variables
        self.df_RAW = df_RAW

        return self