import polars as pl
from .ratingBase import ratingBase

class DIGINETICA(ratingBase):

    def __init__(self):

        # Read
        df_RAW = pl.read_csv("../dataset/DIGINETICA/train-item-views.csv", separator=";", null_values="NA", new_columns=["session_id", "user_id", "item_id", "timeframe", "eventdate"])
        df_RAW = df_RAW.drop("user_id")

        # Cast Datetime
        df_RAW = df_RAW.with_columns( pl.col("eventdate").str.strptime(pl.Datetime, "%Y-%m-%d") )

        # Sort by session_id & timeframe
        df_RAW = df_RAW.sort("session_id", "timeframe")

        # Split into TRAIN, VALID, TEST subset
        df_SUBSET = {
            "TRAIN": df_RAW.filter(pl.col("eventdate").dt.month() < 4),
            "VALID": df_RAW.filter(pl.col("eventdate").dt.month() == 4),
            "TEST": df_RAW.filter(4 < pl.col("eventdate").dt.month()),
        }

        # Make Sequences by session_id
        df_SUBSET = {
            name: df.groupby("session_id", maintain_order=True).agg(pl.col("item_id").alias("item_list")).drop("session_id")
            for name, df in df_SUBSET.items()
        }

        # Delete sessions with a sequence length of 1
        df_SUBSET = {
            name: df.filter( pl.col("item_list").list.lengths() > 1 )
            for name, df in df_SUBSET.items()
        }

        # Reset IDs
        item_ids = pl.concat(df_SUBSET.values()).get_column("item_list").explode().unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(1, len(item_ids)+1)))
        df_SUBSET = {
            name: df.with_columns( pl.col("item_list").list.eval(pl.element().map_dict(item_id_map)) )
            for name, df in df_SUBSET.items()
        }

        # Set Variables
        self.df_RAW = df_RAW
        self.df_SUBSET = df_SUBSET
        self.item_id_map = item_id_map
        self.item_num = len(item_id_map) + 1