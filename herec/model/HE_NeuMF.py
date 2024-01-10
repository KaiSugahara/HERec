import jax
import jax.numpy as jnp
from flax import linen as nn

from .HE import HE

class HE_NeuMF(nn.Module):
    
    user_num: int
    item_num: int
    userClusterNums: list
    itemClusterNums: list
    embedDim: int
    temperature: float
    
    def setup(self):

        self.userGmfEmbedder = HE(
            objNum = self.user_num,
            clusterNums = self.userClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )
        self.itemGmfEmbedder = HE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )
        self.userMlpEmbedder = HE(
            objNum = self.user_num,
            clusterNums = self.userClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )
        self.itemMlpEmbedder = HE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )
        self.Dense1 = nn.Dense(features=(self.embedDim))
        self.Dense2 = nn.Dense(features=(self.embedDim // 2))
        self.DenseL = nn.Dense(features=1)
    
    @nn.compact
    def __call__(self, X):

        user_ids = X[:, 0]
        item_ids = X[:, 1]
        
        # Get Target User/Item Embeddings
        userGmfEmbed = self.userGmfEmbedder.getEmbed(user_ids)
        itemGmfEmbed = self.itemGmfEmbedder.getEmbed(item_ids)
        userMlpEmbed = self.userMlpEmbedder.getEmbed(user_ids)
        itemMlpEmbed = self.itemMlpEmbedder.getEmbed(item_ids)

        # GMF
        X_GMF = userGmfEmbed * itemGmfEmbed

        # MLP
        X_MLP = jnp.hstack([userMlpEmbed, itemMlpEmbed]) # Concatenation
        X_MLP = nn.relu(self.Dense1(X_MLP)) # 1layer -> 2layer
        X_MLP = nn.relu(self.Dense2(X_MLP)) # 2layer -> 3layer

        # NeuMF Layer
        X_LAST = jnp.hstack([X_GMF, X_MLP]) # concatenation
        X_LAST = self.DenseL(X_LAST)

        return X_LAST
    
    def __get_all_scores_by_user(self, userGmfEmbed, userMlpEmbed, itemGmfEmbedMatrix, itemMlpEmbedMatrix):

        # GMF
        X_GMF = userGmfEmbed.reshape(1, -1) * itemGmfEmbedMatrix
        
        # MLP
        X_MLP = jnp.hstack([jnp.repeat(userMlpEmbed.reshape(1, -1), self.item_num, axis=0), itemMlpEmbedMatrix]) # Concatenation
        X_MLP = nn.relu( self.Dense1(X_MLP) ) # 1layer -> 2layer
        X_MLP = nn.relu( self.Dense2(X_MLP) ) # 2layer -> 3layer
        
        # NeuMF Layer
        X_LAST = jnp.hstack([X_GMF, X_MLP]) # concatenation
        X_LAST = self.DenseL(X_LAST)
        
        return X_LAST.reshape(-1)
    
    def get_all_scores_by_user_ids(self, user_ids):
        
        # Get Target User Embeddings
        userGmfEmbedMatrix = self.userGmfEmbedder.getEmbed(user_ids)
        userMlpEmbedMatrix = self.userMlpEmbedder.getEmbed(user_ids)

        # Get All Item Embeddings
        itemGmfEmbedMatrix = self.itemGmfEmbedder.getEmbedByLevel(level=0)
        itemMlpEmbedMatrix = self.itemMlpEmbedder.getEmbedByLevel(level=0)

        return jax.vmap(self.__get_all_scores_by_user, in_axes=(0, 0, None, None), out_axes=0)(userGmfEmbedMatrix, userMlpEmbedMatrix, itemGmfEmbedMatrix, itemMlpEmbedMatrix)