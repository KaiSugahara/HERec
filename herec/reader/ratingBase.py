import polars as pl
from sklearn.model_selection import train_test_split

class ratingBase():

    def __split(self):

        df_SUBSET = {}
        df_SUBSET["TRAIN"], df_SUBSET["TEST"] = train_test_split(self.df_RAW, test_size=0.2, shuffle=False)
        df_SUBSET["TRAIN"], df_SUBSET["VALID"] = train_test_split(df_SUBSET["TRAIN"], test_size=0.2, shuffle=False)

        # Remove Cold Users/Items from VALID & TEST subset
        for name in ["VALID", "TEST"]:
            df_SUBSET[name] = df_SUBSET[name].filter(
                pl.col("user_id").is_in( df_SUBSET["TRAIN"].get_column("user_id").unique() )
                & pl.col("item_id").is_in( df_SUBSET["TRAIN"].get_column("item_id").unique() )
            )
        
        # Reset IDs
        user_ids = pl.concat(df_SUBSET.values()).get_column("user_id").unique(maintain_order=True)
        user_id_map = dict(zip(user_ids, range(len(user_ids))))
        item_ids = pl.concat(df_SUBSET.values()).get_column("item_id").unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(len(item_ids))))
        for name in ["TRAIN", "VALID", "TEST"]:
            df_SUBSET[name] = df_SUBSET[name].with_columns(
                pl.col("user_id").map_dict(user_id_map),
                pl.col("item_id").map_dict(item_id_map),
            )

        # Set Variables
        self.df_SUBSET = df_SUBSET
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.user_num = len(user_id_map)
        self.item_num = len(item_id_map)

        return self

    def __init__(self):

        # READ
        self.read()

        # Split into TRAIN, VALID, TEST subset
        self.__split()