import jax
import jax.numpy as jnp
from flax import linen as nn

class FM(nn.Module):
    
    user_num: int
    item_num: int
    embedDim: int
    
    @nn.compact
    def __call__(self, INPUT):
        
        user_ids = INPUT[:, 0]
        item_ids = INPUT[:, 1]

        # Encode Multi-hot Vector
        X = jnp.hstack([
            jax.nn.one_hot(user_ids, num_classes=self.user_num),
            jax.nn.one_hot(item_ids, num_classes=self.item_num),
        ])

        # Linear Regression Term
        result_X = nn.Dense(features=1)(X).reshape(-1)

        # Interaction Term
        V = self.param(f'latentMatrix', lambda rng: jax.random.normal(rng, ((self.user_num + self.item_num), self.embedDim), jnp.float32))
        interaction_X = jnp.sum((X.dot(V))**2 - (X**2).dot(V**2), axis=1) / 2
        
        return (result_X + interaction_X).reshape(-1, 1)