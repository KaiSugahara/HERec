import jax
import jax.numpy as jnp
from flax import linen as nn

from .DHE import DHE

class DHE_MF(nn.Module):
    
    user_num: int
    item_num: int
    userClusterNums: list
    itemClusterNums: list
    embedDim: int
    temperature: float
    layerNum: int
    
    def setup(self):

        self.userEmbedder = DHE(
            objNum = self.user_num,
            clusterNums = self.userClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
            layerNum = self.layerNum,
        )

        self.itemEmbedder = DHE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
            layerNum = self.layerNum,
        )
    
    @nn.compact
    def __call__(self, X):
        
        user_ids = X[:, 0]
        item_ids = X[:, 1]

        U = self.userEmbedder.getEmbed(user_ids)
        V = self.itemEmbedder.getEmbed(item_ids)
        
        return jnp.sum(U * V, axis=1, keepdims=True)
    
    def get_all_scores_by_user_ids(self, user_ids):

        U = self.userEmbedder.getEmbed(user_ids)
        V = self.itemEmbedder.getEmbedByLevel(level=0)

        return U @ V.T
    
    def regularization_terms(self):
        
        return 0 + self.userEmbedder.regularization_terms() + self.itemEmbedder.regularization_terms()