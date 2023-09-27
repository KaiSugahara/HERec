import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import trange
import random
import jax
import jax.numpy as jnp

class implicitBase():

    def __prepocessingForValidation(self, fold_id):

        # Clone
        df_TRAIN = self._df_SUBSET[fold_id]["TRAIN"].clone()
        df_VALID = self._df_SUBSET[fold_id]["VALID"].clone()

        # Leave First Interaction (Be Unique) for each subset
        df_TRAIN = df_TRAIN.unique(["user_id", "item_id"])
        df_VALID = df_VALID.unique(["user_id", "item_id"])

        # Remove Cold Users/Items from VALID subset
        df_VALID = df_VALID.filter(
            pl.col("user_id").is_in( df_TRAIN.get_column("user_id").unique(maintain_order=True) )
        ).filter(
            pl.col("item_id").is_in( df_TRAIN.get_column("item_id").unique(maintain_order=True) )
        )

        # Reset IDs
        user_ids = pl.concat([df_TRAIN, df_VALID]).get_column("user_id").unique(maintain_order=True)
        user_id_map = dict(zip(user_ids, range(len(user_ids))))
        user_num = len(user_id_map)
        item_ids = pl.concat([df_TRAIN, df_VALID]).get_column("item_id").unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(len(item_ids))))
        item_num = len(item_id_map)
        df_TRAIN = df_TRAIN.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )
        df_VALID = df_VALID.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )

        # Add Positive Item IDs for Each User in TRAIN subset
        df_TRAIN = df_TRAIN.group_by("user_id", maintain_order=True).agg(
            pl.col("item_id"),
            pl.col("item_id").alias("pos_item_ids"),
        ).explode("item_id")

        # Add Positive Item IDs for Each User in VALID subset
        df_VALID = df_VALID.group_by("user_id", maintain_order=True).agg(
            pl.col("item_id").alias("pos_item_ids"),
        )

        # Fill -1 to pos_item_ids such that it has the same length in VALID subset
        max_len = df_VALID.get_column("pos_item_ids").list.lengths().max()
        df_VALID = df_VALID.with_columns(
            pl.col("pos_item_ids").list.concat([-1]*max_len).list.head(max_len)
        )

        # Convert TEST subset for Evaluation
        df_VALID = dict(
            # User IDs
            user_ids = jax.device_put(df_VALID.get_column("user_id").to_numpy()),
            # True Item IDs
            true_items = jnp.array(df_VALID.get_column("pos_item_ids").to_list()),
            # Length of True Items by User
            true_item_len = jnp.array(df_VALID.get_column("pos_item_ids").list.set_difference(-1).list.lengths().to_list()),
        )

        # Set Variables
        self.VALIDATION[fold_id] = {
            "df_TRAIN": df_TRAIN,
            "df_VALID": df_VALID,
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "user_num": user_num,
            "item_num": item_num,
        }

        return self
    
    def __prepocessingForTest(self, fold_id):

        # Clone
        df_TRAIN = pl.concat([ self._df_SUBSET[fold_id]["TRAIN"], self._df_SUBSET[fold_id]["VALID"] ])
        df_TEST = self._df_SUBSET[fold_id]["TEST"].clone()

        # Leave First Interaction (Be Unique) for each subset
        df_TRAIN = df_TRAIN.unique(["user_id", "item_id"])
        df_TEST = df_TEST.unique(["user_id", "item_id"])

        # Remove Cold Users/Items from TEST subset
        df_TEST = df_TEST.filter(
            pl.col("user_id").is_in( df_TRAIN.get_column("user_id").unique(maintain_order=True) )
        ).filter(
            pl.col("item_id").is_in( df_TRAIN.get_column("item_id").unique(maintain_order=True) )
        )

        # Reset IDs
        user_ids = pl.concat([df_TRAIN, df_TEST]).get_column("user_id").unique(maintain_order=True)
        user_id_map = dict(zip(user_ids, range(len(user_ids))))
        user_num = len(user_id_map)
        item_ids = pl.concat([df_TRAIN, df_TEST]).get_column("item_id").unique(maintain_order=True)
        item_id_map = dict(zip(item_ids, range(len(item_ids))))
        item_num = len(item_id_map)
        df_TRAIN = df_TRAIN.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )
        df_TEST = df_TEST.with_columns(
            pl.col("user_id").map_dict(user_id_map),
            pl.col("item_id").map_dict(item_id_map),
        )

        # Add Positive Item IDs for Each User in TRAIN subset
        df_TRAIN = df_TRAIN.group_by("user_id", maintain_order=True).agg(
            pl.col("item_id"),
            pl.col("item_id").alias("pos_item_ids"),
        ).explode("item_id")

        # Add Positive Item IDs for Each User in TEST subset
        df_TEST = df_TEST.group_by("user_id", maintain_order=True).agg(
            pl.col("item_id").alias("pos_item_ids"),
        )

        # Fill -1 to pos_item_ids such that it has the same length in TEST subset
        max_len = df_TEST.get_column("pos_item_ids").list.lengths().max()
        df_TEST = df_TEST.with_columns(
            pl.col("pos_item_ids").list.concat([-1]*max_len).list.head(max_len)
        )

        # Convert TEST subset for Evaluation
        df_TEST = dict(
            # User IDs
            user_ids = jax.device_put(df_TEST.get_column("user_id").to_numpy()),
            # True Item IDs
            true_items = jnp.array(df_TEST.get_column("pos_item_ids").to_list()),
            # Length of True Items by User
            true_item_len = jnp.array(df_TEST.get_column("pos_item_ids").list.set_difference(-1).list.lengths().to_list()),
        )

        # Set Variables
        self.TEST[fold_id] = {
            "df_TRAIN": df_TRAIN,
            "df_TEST": df_TEST,
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "user_num": user_num,
            "item_num": item_num,
        }

        return self

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
            for fold_id in [0, 1, 2]
        }

        return self

    def __init__(self):

        # READ
        self.read()

        # Split into TRAIN, VALID, TEST subset
        self.__split()

        # Preprocessing
        self.VALIDATION = {} # Initialize
        self.TEST = {} # Initialize
        for fold_id in [0, 1, 2]:
            self.__prepocessingForValidation(fold_id)
            self.__prepocessingForTest(fold_id)