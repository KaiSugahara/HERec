import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np

class implicitBase():

    def __prepocessingForValidation(self, fold_id):

        # Clone
        df_TRAIN = self._df_SUBSET[fold_id]["TRAIN"].clone()
        df_VALID = self._df_SUBSET[fold_id]["VALID"].clone()

        # Remove Cold Users/Items from VALID subset
        df_VALID = df_VALID.filter(
            pl.col("user_id").is_in( df_TRAIN.get_column("user_id").unique() )
        ).filter(
            pl.col("item_id").is_in( df_TRAIN.get_column("item_id").unique() )
        )

        # Reset IDs
        user_ids = pl.concat([df_TRAIN, df_VALID]).get_column("user_id").unique(maintain_order=True)
        user_id_map = dict(zip(user_ids, range(len(user_ids))))
        item_ids = pl.concat([df_TRAIN, df_VALID]).get_column("item_id").unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(len(item_ids))))
        df_TRAIN = df_TRAIN.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )
        df_VALID = df_VALID.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )

        # Set Variables
        self.VALIDATION[fold_id] = {
            "df_TRAIN": df_TRAIN,
            "df_VALID": df_VALID,
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "user_num": len(user_id_map),
            "item_num": len(item_id_map),
        }

        return self
    
    def __prepocessingForTest(self, fold_id):

        # Clone
        df_TRAIN = pl.concat([ self._df_SUBSET[fold_id]["TRAIN"], self._df_SUBSET[fold_id]["VALID"] ])
        df_TEST = self._df_SUBSET[fold_id]["TEST"].clone()

        # Remove Cold Users/Items from TEST subset
        df_TEST = df_TEST.filter(
            pl.col("user_id").is_in( df_TRAIN.get_column("user_id").unique() )
        ).filter(
            pl.col("item_id").is_in( df_TRAIN.get_column("item_id").unique() )
        )

        # Reset IDs
        user_ids = pl.concat([df_TEST, df_TEST]).get_column("user_id").unique(maintain_order=True)
        user_id_map = dict(zip(user_ids, range(len(user_ids))))
        item_ids = pl.concat([df_TEST, df_TEST]).get_column("item_id").unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(len(item_ids))))
        df_TRAIN = df_TRAIN.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )
        df_TEST = df_TEST.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )

        # Set Variables
        self.TEST[fold_id] = {
            "df_TRAIN": df_TRAIN,
            "df_TEST": df_TEST,
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "user_num": len(user_id_map),
            "item_num": len(item_id_map),
        }

        return self

    def __split(self):

        # Initialize
        self.VALIDATION = {}
        self.TEST = {}

        # Split Data Indices to 10 subsets
        subset_indices = np.array_split(range(self.df_RAW.height), 10)

        # Split Data Indices to 3 folds under temporal splitting and CV
        self._df_SUBSET = {
            fold_id: {
                "TRAIN": self.df_RAW[np.concatenate(subset_indices[fold_id:fold_id+6])],
                "VALID": self.df_RAW[subset_indices[fold_id+6]],
                "TEST": self.df_RAW[subset_indices[fold_id+7]],
            }
            for fold_id in [0, 1, 2]
        }

        # Preprocessing
        for fold_id in [0, 1, 2]:
            self.__prepocessingForValidation(fold_id)
            self.__prepocessingForTest(fold_id)

        return self

    def __init__(self):

        # READ
        self.read()

        # Split into TRAIN, VALID, TEST subset
        self.__split()