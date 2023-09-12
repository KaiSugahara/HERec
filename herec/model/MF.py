import jax
import jax.numpy as jnp
from flax import linen as nn

class MF(nn.Module):
    
    user_num: int
    item_num: int
    embedDim: int
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)(user_ids)
        V = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)