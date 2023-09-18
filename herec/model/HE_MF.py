import jax
import jax.numpy as jnp
from flax import linen as nn

from .HE import HE

class HE_MF(nn.Module):
    
    user_num: int
    item_num: int
    userClusterNums: list
    itemClusterNums: list
    embedDim: int
    temp: float
    
    def setup(self):

        self.userEmbedder = HE(
            objNum = self.user_num,
            clusterNums = self.userClusterNums,
            embedDim = self.embedDim,
            temp = self.temp,
        )

        self.itemEmbedder = HE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            temp = self.temp,
        )
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = self.userEmbedder.getEmbed(user_ids)
        V = self.itemEmbedder.getEmbed(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)