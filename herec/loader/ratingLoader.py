import jax
import math

class ratingLoader:

    def __init__(self, key, df_DATA, batch_size):
        
        # Set batch_size
        self.batch_size = batch_size
        
        # Extract Data Size
        self.data_size = df_DATA.height
        
        # Calc Batch #
        self.batch_num = math.ceil(self.data_size / self.batch_size)
        
        # Make Shuffled Indices of Data
        self.shuffled_indices = jax.random.permutation(key, self.data_size)

        # Split Data into Input Matrix X and Label Matrix Y
        self.X = jax.device_put(df_DATA.select("user_id", "item_id").to_numpy())
        self.Y = jax.device_put(df_DATA.select("rating").to_numpy())

        # Shuffle rows of X and Y
        self.X = self.X[self.shuffled_indices]
        self.Y = self.Y[self.shuffled_indices]

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
            Y = jax.lax.dynamic_slice_in_dim(self.Y, start_index, slice_size)
            
            # Update batch-index
            self.batch_idx += 1
            
            return X, Y