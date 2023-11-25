import jax
import math
import polars as pl

class bprLoader:
    
    def add_neg_item_id(self, df):

        # Negative Sampling: Initialize
        column_name = f"neg_item_id"
        df = df.with_columns( pl.lit(-1).alias(column_name), dup_num=True )

        step = 0
        
        while df.get_column("dup_num").any():
        
            # Sampling
            df = df.with_columns(
                pl.when( pl.col("dup_num") )
                .then( pl.col("item_id").sample(df.height, with_replacement=True, shuffle=True, seed=step).alias( column_name ) )
                .otherwise( pl.col(column_name) )
            )
            # Check Duplicates between Positive and Negative Items
            df = df.with_columns(
                pl.when( pl.col("dup_num") )
                .then( pl.col("pos_item_ids").list.contains(pl.col(column_name)).alias("dup_num") )
                .otherwise( pl.col("dup_num") )
            )

            step += 1

        return df.select("user_id", "item_id", "neg_item_id")

    def __init__(self, key, df_DATA, batch_size):
        
        # Set batch_size
        self.batch_size = batch_size
        
        # Extract Data Size
        self.data_size = df_DATA.height
        
        # Calc Batch #
        self.batch_num = math.ceil(self.data_size / self.batch_size)
        
        # Make Shuffled Indices of Data
        self.shuffled_indices = jax.random.permutation(key, self.data_size)

        # Shuffle
        df_X = df_DATA.sample(df_DATA.height, shuffle=True, seed=key[0].tolist())

        # Split to MiniBatch with Negative Sampling
        self.X_list = [
            jax.device_put( self.add_neg_item_id(df_X[start_idx:(start_idx+batch_size)]).to_numpy() )
            for start_idx in range(0, df_X.height, batch_size)
        ]

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
            X = self.X_list[self.batch_idx]
            
            # Update batch-index
            self.batch_idx += 1
            
            return X, None