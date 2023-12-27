import polars as pl
import numpy as np

import jax
import jax.numpy as jnp

class sessionBase():

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

        # Reset IDs
        item_ids = pl.concat([df_TRAIN, df_EVALUATION]).get_column("item_list").explode().unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(1, len(item_ids)+1)))
        item_num = len(item_id_map)
        df_TRAIN = df_TRAIN.with_columns(
            pl.col("item_list").list.eval( pl.element().map_dict(item_id_map) ),
        )
        df_EVALUATION = df_EVALUATION.with_columns(
            pl.col("item_list").list.eval( pl.element().map_dict(item_id_map) ),
        )
        
        # Sort Rows for Efficiently Evaluation
        df_EVALUATION = df_EVALUATION.sort( pl.col("item_list").list.len() )

        return {
            "df_TRAIN": df_TRAIN,
            "df_EVALUATION": df_EVALUATION,
            "user_id_map": None,
            "item_id_map": item_id_map,
            "user_num": None,
            "item_num": item_num,
        }

    def __split(self):

        # Split Data Indices to 10 subsets
        timestamp_subsets = np.array_split( self.df_RAW.get_column("timestamp").unique().sort(), 10 )

        # Split Data Indices to 3 folds under temporal splitting and CV
        self._df_SUBSET = {
            fold_id: {
                "TRAIN": self.df_RAW.filter( pl.col("timestamp").is_in( np.hstack(timestamp_subsets[fold_id:fold_id+6]).tolist() ) ),
                "VALID": self.df_RAW.filter( pl.col("timestamp").is_in( timestamp_subsets[fold_id+6].tolist() ) ),
                "TEST": self.df_RAW.filter( pl.col("timestamp").is_in( timestamp_subsets[fold_id+7].tolist() ) ),
            }
            for fold_id in range(3)
        }

        return self

    def __init__(self):

        # READ
        self.read()

        # Split into TRAIN, VALID, TEST subset
        self.__split()