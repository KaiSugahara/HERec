import jax
import jax.numpy as jnp
from flax import linen as nn

class FM(nn.Module):
    
    user_num: int
    item_num: int
    embedDim: int
    
    def setup(self):
        
        self.userBias = nn.Embed(num_embeddings=self.user_num, features=1)
        self.itemBias = nn.Embed(num_embeddings=self.item_num, features=1)
        self.userEmbedder = nn.Embed(num_embeddings=self.user_num, features=1)
        self.itemEmbedder = nn.Embed(num_embeddings=self.item_num, features=1)
    
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
        interactionTerm = jnp.sum( self.userEmbedder(user_ids) * self.itemEmbedder(item_ids), axis=1, keepdims=True )
        
        return w0 + userLinearTerm + itemLinearTerm + interactionTerm