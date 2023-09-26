import polars as pl
from .implicitBase import implicitBase
from herec.utils import *

class DIGINETICA(implicitBase):

    def read(self):

        # READ
        df_RAW = pl.read_csv(f"{getRepositoryPath()}/dataset/DIGINETICA/train-item-views.csv", separator=";", null_values="NA", new_columns=["session_id", "user_id", "item_id", "timeframe", "eventdate"])

        # Cast Datetime
        df_RAW = df_RAW.with_columns( pl.col("eventdate").str.to_date("%Y-%m-%d") )

        # Clean User ID
        df_RAW = df_RAW.group_by("session_id").agg( pl.all() ).with_columns(
            pl.col("user_id").list.eval(pl.element().drop_nulls()).list.unique()
        ).filter(
            pl.col("user_id").list.lengths() > 0
        ).explode(
            pl.col("user_id")
        ).explode(
            "item_id", "timeframe", "eventdate"
        )

        # Sort
        df_RAW = df_RAW.sort("eventdate", "session_id", "timeframe")

        # Select Columns
        df_RAW = df_RAW.select( pl.col("user_id"), pl.col("item_id"), pl.col("eventdate").alias("timestamp") )

        # Set Variables
        self.df_RAW = df_RAW

        return self