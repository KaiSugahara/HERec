import jax
import jax.numpy as jnp
from flax import linen as nn

class MF(nn.Module):
    
    user_num: int
    item_num: int
    embedDim: int

    def setup(self):

        self.userEmbedder = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)
        self.itemEmbedder = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = self.userEmbedder(user_ids)
        V = self.itemEmbedder(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)

    def get_all_scores_by_user_ids(self, user_ids):

        U = self.userEmbedder(user_ids)
        V = self.itemEmbedder.embedding

        return U @ V.T