import jax
import jax.numpy as jnp
from flax import linen as nn

from .HE import HE

class HE_FM(nn.Module):
    
    user_num: int
    item_num: int
    userClusterNums: list
    itemClusterNums: list
    temperature: float
    embedDim: int

    def setup(self):
        
        self.userBias = nn.Embed(num_embeddings=self.user_num, features=1)
        self.itemBias = nn.Embed(num_embeddings=self.item_num, features=1)

        self.userEmbedder = HE(
            objNum = self.user_num,
            clusterNums = self.userClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )

        self.itemEmbedder = HE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )
    
    @nn.compact
    def __call__(self, INPUT):
        
        user_ids = INPUT[:, 0]
        item_ids = INPUT[:, 1]
        
        # Bias Term
        w0 = self.param(f'biasTerm', lambda rng: jax.random.normal(rng, (1, 1), jnp.float32))

        # Linear Regression Term
        userLinearTerm = self.userBias(user_ids)
        itemLinearTerm = self.itemBias(item_ids)

        # Interaction Term
        interactionTerm = jnp.sum( self.userEmbedder.getEmbed(user_ids) * self.itemEmbedder.getEmbed(item_ids), axis=1, keepdims=True )
        
        return w0 + userLinearTerm + itemLinearTerm + interactionTerm