import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from herec.utils import *

class AMAZON_M2():

    def __init__(self):

        # READ
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

        # Split into TRAIN, VALID, TEST subset
        df_SUBSET = {}
        df_SUBSET["TRAIN"], df_SUBSET["TEST"] = train_test_split(df_RAW, test_size=0.2, shuffle=True, random_state=0)
        df_SUBSET["TRAIN"], df_SUBSET["VALID"] = train_test_split(df_SUBSET["TRAIN"], test_size=0.2, shuffle=True, random_state=0)

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