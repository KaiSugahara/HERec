import jax
import jax.numpy as jnp
from flax import linen as nn

from .HE import HE

class HE_MF_ITEM(nn.Module):
    
    user_num: int
    item_num: int
    itemClusterNums: list
    embedDim: int
    lam_exc: float
    lam_inc: float
    
    def setup(self):

        self.userEmbedder = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)

        self.itemEmbedder = HE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            lam_exc = self.lam_exc,
            lam_inc = self.lam_inc,
        )
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = self.userEmbedder(user_ids)
        V = self.itemEmbedder.getEmbed(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)
    
    def get_all_scores_by_user_ids(self, user_ids):

        U = self.userEmbedder(user_ids)
        V = self.itemEmbedder.getEmbedByLevel(level=0)

        return U @ V.T
    
    def regularization_terms(self):
        
        return 0 + self.itemEmbedder.regularization_terms()