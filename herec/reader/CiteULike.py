import polars as pl
from .implicitBase import implicitBase
from ..utils.getRepositoryPath import getRepositoryPath

class CiteULike(implicitBase):

    def read(self):

        # Read
        df_RAW = pl.read_csv("https://raw.githubusercontent.com/js05212/citeulike-a/master/users.dat", has_header=False)
        
        # Make User ID and Item ID
        df_RAW = df_RAW.select(
            pl.arange(0, df_RAW.height).alias("user_id"),
            pl.col("column_1").str.split(" ").list.slice(1).alias("item_id")
        )
        df_RAW = df_RAW.explode( pl.col("item_id") )
        
        # Shuffle interactions and Add dummy timestamp
        df_RAW = df_RAW.sample(fraction=1, shuffle=True, seed=0)
        df_RAW = df_RAW.with_columns( pl.col("item_id").cast(int), pl.arange(0, df_RAW.height).alias("timestamp") )

        # Set Variables
        self.df_RAW = df_RAW

        return self