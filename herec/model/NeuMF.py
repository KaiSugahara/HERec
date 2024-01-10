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
        userGmfEmbedMatrix = self.userGmfEmbedder(user_ids)
        userMlpEmbedMatrix = self.userMlpEmbedder(user_ids)

        # Get All Item Embeddings
        itemGmfEmbedMatrix = self.itemGmfEmbedder.embedding
        itemMlpEmbedMatrix = self.itemMlpEmbedder.embedding

        return jax.vmap(self.__get_all_scores_by_user, in_axes=(0, 0, None, None), out_axes=0)(userGmfEmbedMatrix, userMlpEmbedMatrix, itemGmfEmbedMatrix, itemMlpEmbedMatrix)