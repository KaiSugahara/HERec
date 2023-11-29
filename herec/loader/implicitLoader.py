import jax
import math
import polars as pl

class implicitLoader:
    
    n_neg: int
    has_weight: bool
    sampler: str

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

        # Extract User Num.
        item_num = df_X["item_id"].n_unique()

        # Extract Positive Items
        df_X = df_X.group_by("user_id", maintain_order=True).agg("item_id", "pop_weight").with_columns( pl.col("item_id").list.unique().alias("pos_item_ids") ).explode("item_id", "pop_weight")

        # Negative Sampling
        for i in range(self.n_neg):

            # Initialize
            column_name = f"neg_item_id_{i}"
            df_X = df_X.with_columns( pl.lit(-1).alias(column_name), dup_num=True )

            while df_X.get_column("dup_num").any():
                
                # Generate Key for sampling
                key, subkey = jax.random.split(key)
                # Sampling
                if self.sampler == "random":
                    df_X = df_X.with_columns(
                        pl.when( pl.col("dup_num") )
                        .then( pl.arange(0, item_num).sample(self.data_size, shuffle=True, with_replacement=True, seed=subkey.tolist()[0]).alias(column_name) )
                        .otherwise( pl.col(column_name) )
                    )
                elif self.sampler == "pop":
                    df_X = df_X.with_columns(
                        pl.when( pl.col("dup_num") )
                        .then( pl.col("item_id").sample(self.data_size, shuffle=True, with_replacement=True, seed=subkey.tolist()[0]).alias(column_name) )
                        .otherwise( pl.col(column_name) )
                    )
                # Check Duplicates between Positive and Negative Items
                df_X = df_X.with_columns(
                    pl.when( pl.col("dup_num") )
                    .then( pl.col("pos_item_ids").list.contains(pl.col(column_name)).alias("dup_num") )
                    .otherwise( pl.col("dup_num") )
                )

        # Convert to Matrix
        self.X = jax.device_put(df_X.select("user_id", "item_id", "^neg_item_id_\d+$").to_numpy())
        self.Y = jax.device_put(df_X.select("pop_weight").to_numpy()) if self.has_weight else None
        
        # Shuffle rows of X
        self.X = self.X[self.shuffled_indices]
        self.Y = self.Y[self.shuffled_indices] if self.has_weight else None

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
            end_index = self.batch_size * (self.batch_idx + 1)
            X = self.X[start_index:end_index]
            Y = self.Y[start_index:end_index] if self.has_weight else None
            
            # Update batch-index
            self.batch_idx += 1
            
            return X, Y