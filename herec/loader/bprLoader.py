import jax
import math
import polars as pl

class bprLoader:

    def __init__(self, key, df_DATA, batch_size):
        
        # Set batch_size
        self.batch_size = batch_size
        
        # Extract Data Size
        self.data_size = df_DATA.height
        
        # Calc Batch #
        self.batch_num = math.ceil(self.data_size / self.batch_size)
        
        # Make Shuffled Indices of Data
        self.shuffled_indices = jax.random.permutation(key, self.data_size)

        # Clone
        df_X = df_DATA.clone()

        # Extract Item Num.
        item_num = df_X["item_id"].n_unique()

        # Negative Sampling: Initialize
        column_name = f"neg_item_id"
        df_X = df_X.with_columns( pl.lit(-1).alias(column_name), dup_num=True )

        while df_X.get_column("dup_num").any():
            
            # Sampling
            df_X = df_X.with_columns(
                pl.when( pl.col("dup_num") )
                .then( pl.arange(0, item_num).sample(self.data_size, with_replacement=True).alias(column_name) )
                .otherwise( pl.col(column_name) )
            )
            # Check Duplicates between Positive and Negative Items
            df_X = df_X.with_columns(
                pl.when( pl.col("dup_num") )
                .then( pl.col("pos_item_ids").list.contains(pl.col(column_name)).alias("dup_num") )
                .otherwise( pl.col("dup_num") )
            )

        # Convert to Matrix
        self.X = jax.device_put(df_X.select("user_id", "item_id", "neg_item_id").to_numpy())
        
        # Shuffle rows of X
        self.X = self.X[self.shuffled_indices]

    def __iter__(self):
        
        # Initialize batch-index as 0
        self.batch_idx = 0
        
        return self

    def __next__(self):
        
        if self.batch_idx == self.batch_num:
            
            # Stop
            raise StopIteration()
            
        else:
            
            # Extract {batch_idx}-th Minibatch
            start_index = self.batch_size * self.batch_idx
            slice_size = min( self.batch_size, (self.data_size - start_index) )
            X = jax.lax.dynamic_slice_in_dim(self.X, start_index, slice_size)
            
            # Update batch-index
            self.batch_idx += 1
            
            return X, None