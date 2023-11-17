import pandas as pd
import polars as pl
from .sessionBase import sessionBase
from herec.utils import *

class AMAZON_M2(sessionBase):

    def read(self):

        # Read
        df_RAW = pd.read_csv(f"{getRepositoryPath()}/dataset/AMAZON_M2/sessions_train.csv")
        df_RAW = pl.from_pandas(df_RAW)

        # Filter sessions in JP
        df_RAW = df_RAW.filter( pl.col("locale") == "JP" )

        # str2list
        df_RAW = df_RAW.with_columns(
            pl.col("prev_items")
            .str.replace_all('[', '', literal=True)
            .str.replace_all(']', '', literal=True)
            .str.replace_all("'", '', literal=True)
            .str.replace_all('\n', ' ', literal=True)
            .str.replace_all('\r', ' ', literal=True)
            .str.replace_all('  ', ' ', literal=True)
            .str.split(" ")
        )

        # Make Sequences
        df_RAW = df_RAW.select(
            pl.concat_list( "prev_items", "next_item" ).alias("item_list")
        )
        
        # Add Dummy Timestamp
        df_RAW = df_RAW.with_columns(
            pl.arange(0, df_RAW.height).alias("timestamp")
        )

        # Set Variables
        self.df_RAW = df_RAW

        return self