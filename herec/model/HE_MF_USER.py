import jax
import jax.numpy as jnp
from flax import linen as nn

from .HE import HE

class HE_MF_USER(nn.Module):
    
    user_num: int
    item_num: int
    userClusterNums: list
    embedDim: int
    lam_exc: float
    lam_inc: float
    
    def setup(self):

        self.userEmbedder = HE(
            objNum = self.user_num,
            clusterNums = self.userClusterNums,
            embedDim = self.embedDim,
            lam_exc = self.lam_exc,
            lam_inc = self.lam_inc,
        )

        self.itemEmbedder = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = self.userEmbedder.getEmbed(user_ids)
        V = self.itemEmbedder(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)
    
    def get_all_scores_by_user_ids(self, user_ids):

        U = self.userEmbedder.getEmbed(user_ids)
        V = self.itemEmbedder.embedding

        return U @ V.T
    
    def regularization_terms(self):
        
        return 0 + self.userEmbedder.regularization_terms()