import polars as pl
import numpy as np

class ratingBase():

    def get(self, fold_id, work):

        # Clone
        if work == "train":
            df_TRAIN = self._df_SUBSET[fold_id]["TRAIN"].clone()
            df_EVALUATION = self._df_SUBSET[fold_id]["VALID"].clone()
        elif work == "test":
            df_TRAIN = pl.concat([ self._df_SUBSET[fold_id]["TRAIN"], self._df_SUBSET[fold_id]["VALID"] ])
            df_EVALUATION = self._df_SUBSET[fold_id]["TEST"].clone()
        else:
            raise Exception()

        # Remove Cold Users/Items from VALID subset
        df_EVALUATION = df_EVALUATION.filter(
            pl.col("user_id").is_in( df_TRAIN.get_column("user_id").unique() )
        ).filter(
            pl.col("item_id").is_in( df_TRAIN.get_column("item_id").unique() )
        )

        # Reset IDs
        user_ids = pl.concat([df_TRAIN, df_EVALUATION]).get_column("user_id").unique(maintain_order=True)
        user_id_map = dict(zip(user_ids, range(len(user_ids))))
        user_num = len(user_id_map)
        item_ids = pl.concat([df_TRAIN, df_EVALUATION]).get_column("item_id").unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(len(item_ids))))
        item_num = len(item_id_map)
        df_TRAIN = df_TRAIN.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )
        df_EVALUATION = df_EVALUATION.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )

        return {
            "df_TRAIN": df_TRAIN,
            "df_EVALUATION": df_EVALUATION,
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "user_num": user_num,
            "item_num": item_num,
        }

    def __split(self):

        # Split Data Indices to 10 subsets
        timestamp_subsets = np.array_split( self.df_RAW.get_column("timestamp").unique().sort(), 10 )

        # Split Data Indices to 5 folds under temporal splitting and CV
        self._df_SUBSET = {
            fold_id: {
                "TRAIN": self.df_RAW.filter( pl.col("timestamp").is_in( np.hstack(timestamp_subsets[fold_id:fold_id+4]).tolist() ) ),
                "VALID": self.df_RAW.filter( pl.col("timestamp").is_in( timestamp_subsets[fold_id+4].tolist() ) ),
                "TEST": self.df_RAW.filter( pl.col("timestamp").is_in( timestamp_subsets[fold_id+5].tolist() ) ),
            }
            for fold_id in range(5)
        }

        return self

    def __init__(self):

        # READ
        self.read()

        # Split into TRAIN, VALID, TEST subset
        self.__split()