import jax
import jax.numpy as jnp
from flax import linen as nn

class ProtoMF(nn.Module):
    
    user_num: int
    item_num: int
    user_proto_num: int
    item_proto_num: int
    embedDim: int
    lam1: float
    lam2: float
    lam3: float
    lam4: float
    
    def setup(self):

        self.userEmbed = nn.Embed(num_embeddings=self.user_num, features=self.embedDim)
        self.itemEmbed = nn.Embed(num_embeddings=self.item_num, features=self.embedDim)
        self.userProtoEmbed = nn.Embed(num_embeddings=self.user_proto_num, features=self.embedDim)
        self.itemProtoEmbed = nn.Embed(num_embeddings=self.item_proto_num, features=self.embedDim)
        self.DenseItem = nn.Dense(features=self.user_proto_num)
        self.DenseUser = nn.Dense(features=self.item_proto_num)
    
    @nn.compact
    def __call__(self, X):

        user_ids = X[:, 0]
        item_ids = X[:, 1]

        # U-Score (for target items)
        U_new = self.get_user_new_embeddings(user_ids)
        T = self.itemEmbed(item_ids)
        T_dummy = self.DenseItem(T)
        U_score = jnp.sum(U_new * T_dummy, axis=1, keepdims=True)

        # I-Score (for target items)
        T_new = self.get_item_new_embeddings(item_ids)
        U = self.userEmbed(user_ids)
        U_dummy = self.DenseUser(U)
        I_score = jnp.sum(T_new * U_dummy, axis=1, keepdims=True)

        # UI-Score (for target items)
        UI_Score = U_score + I_score
        
        return UI_Score
    
    def get_all_scores_by_user_ids(self, user_ids):

        # U-Score (for all items)
        U_new = self.get_user_new_embeddings(user_ids)
        T = self.itemEmbed.embedding
        T_dummy = self.DenseItem(T)
        U_score = U_new @ T_dummy.T

        # I-Score (for all items)
        T_new = self.get_item_new_embeddings(jnp.arange(0, self.item_num))
        U = self.userEmbed(user_ids)
        U_dummy = self.DenseUser(U)
        I_score = U_dummy @ T_new.T

        # UI-Score (for all items)
        UI_Score = U_score + I_score

        return UI_Score

    def get_user_new_embeddings(self, ids):

        # User Embedding(s)
        U = self.userEmbed(ids)

        # User Proto Embedding(s)
        U_proto = self.userProtoEmbed.embedding
        
        # Generate User New Embedding(s)
        U_new = U @ U_proto.T
        U_new = U_new / jnp.linalg.norm(U_proto, ord=2, axis=1)
        U_new = (U_new.T / jnp.linalg.norm(U, ord=2, axis=1)).T
        U_new = U_new + 1

        return U_new

    def get_item_new_embeddings(self, ids):

        # Item Embedding(s)
        T = self.itemEmbed(ids)

        # Item Proto Embedding(s)
        T_proto = self.itemProtoEmbed.embedding
        
        # Generate Item New Embedding(s)
        T_new = T @ T_proto.T
        T_new = T_new / jnp.linalg.norm(T_proto, ord=2, axis=1)
        T_new = (T_new.T / jnp.linalg.norm(T, ord=2, axis=1)).T
        T_new = T_new + 1

        return T_new
    
    def regularization_terms(self):
        
        loss = 0
        
        U_new = self.get_user_new_embeddings(jnp.arange(0, self.user_num))
        T_new = self.get_item_new_embeddings(jnp.arange(0, self.item_num))
        
        loss -= self.lam1 * U_new.max(axis=0).mean()
        loss -= self.lam2 * U_new.max(axis=1).mean()
        loss -= self.lam3 * T_new.max(axis=0).mean()
        loss -= self.lam4 * T_new.max(axis=1).mean()
        
        return loss