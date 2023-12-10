import jax
import jax.numpy as jnp
from flax import linen as nn

class NeuMF(nn.Module):
    
    user_num: int
    item_num: int
    embedDim: int
    
    def setup(self):

        self.userGmfEmbedder = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)
        self.itemGmfEmbedder = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)
        self.userMlpEmbedder = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)
        self.itemMlpEmbedder = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)
        self.Dense1 = nn.Dense(features=(self.embedDim))
        self.Dense2 = nn.Dense(features=(self.embedDim // 2))
        self.DenseL = nn.Dense(features=1)
    
    @nn.compact
    def __call__(self, X):

        user_ids = X[:, 0]
        item_ids = X[:, 1]
        
        # Get Target User/Item Embeddings
        userGmfEmbed = self.userGmfEmbedder(user_ids)
        itemGmfEmbed = self.itemGmfEmbedder(item_ids)
        userMlpEmbed = self.userMlpEmbedder(user_ids)
        itemMlpEmbed = self.itemMlpEmbedder(item_ids)

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
    
    def __get_all_scores_by_user_id(self, user_id, itemGmfEmbed, itemMlpEmbed):

        # Get Target User Embeddings
        userGmfEmbed = jnp.tile( self.userGmfEmbedder(user_id), (self.item_num, 1) )
        userMlpEmbed = jnp.tile( self.userMlpEmbedder(user_id), (self.item_num, 1) )

        # GMF
        X_GMF = userGmfEmbed * itemGmfEmbed
        
        # MLP
        X_MLP = jnp.hstack([userMlpEmbed, itemMlpEmbed]) # Concatenation
        X_MLP = nn.relu( self.Dense1(X_MLP) ) # 1layer -> 2layer
        X_MLP = nn.relu( self.Dense2(X_MLP) ) # 2layer -> 3layer
        
        # NeuMF Layer
        X_LAST = jnp.hstack([X_GMF, X_MLP]) # concatenation
        X_LAST = self.DenseL(X_LAST)
        
        return X_LAST.reshape(-1)
    
    def get_all_scores_by_user_ids(self, user_ids):

        # Get All Item Embeddings
        itemGmfEmbed = self.itemGmfEmbedder.embedding
        itemMlpEmbed = self.itemMlpEmbedder.embedding

        return jax.vmap(self.__get_all_scores_by_user_id, in_axes=(0, None, None), out_axes=0)(user_ids, itemGmfEmbed, itemMlpEmbed)