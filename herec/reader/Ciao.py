import polars as pl
import scipy.io
from .ratingBase import ratingBase
from ..utils.getRepositoryPath import getRepositoryPath

class Ciao(ratingBase):

    def read(self):

        # Read
        df_RAW = scipy.io.loadmat(f"{getRepositoryPath()}/dataset/Ciao/ciao_with_rating_timestamp/rating_with_timestamp.mat")
        df_RAW = pl.from_numpy(df_RAW["rating"])
        df_RAW.columns = ["user_id", "item_id", "category", "rating", "helpfulness", "timestamp"]
        df_RAW = df_RAW.sort("timestamp")

        # Set Variables
        self.df_RAW = df_RAW

        return self