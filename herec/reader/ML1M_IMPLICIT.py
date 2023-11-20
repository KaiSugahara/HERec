import polars as pl
from .implicitBase import implicitBase
from ..utils.getRepositoryPath import getRepositoryPath

class ML1M_IMPLICIT(implicitBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/ML1M/ml-1m/ratings.dat", separator=":", has_header=False)
        df_RAW = df_RAW[:, 0::2]
        df_RAW.columns = ["user_id", "item_id", "rating", "timestamp"]
        df_RAW = df_RAW.sort("timestamp")
        
        # Keep only highly rated interactions
        df_RAW = df_RAW.filter( pl.col("rating") > 3.5 )

        # Set Variables
        self.df_RAW = df_RAW

        return self