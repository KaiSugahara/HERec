import jax
import jax.numpy as jnp
from flax import linen as nn

class MF(nn.Module):
    
    user_num: int
    item_num: int
    embedDim: int

    def setup(self):

        self.userEmbed = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)
        self.itemEmbed = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = self.userEmbed(user_ids)
        V = self.itemEmbed(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)

    def get_all_scores_by_user_ids(self, user_ids):

        U = self.userEmbed(user_ids)
        V = self.itemEmbed.embedding

        return U @ V.T
    
    def regularization_terms(self):
        
        return 0