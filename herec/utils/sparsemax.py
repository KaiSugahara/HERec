import jax
import jax.numpy as jnp

def sparsemax(x):

    idxs = (jnp.arange(x.shape[1]) + 1).reshape(1, -1)
    sorted_x = jnp.flip(jax.lax.sort(x, dimension=1), axis=1)
    cum = jnp.cumsum(sorted_x, axis=1)
    k = jnp.sum(jnp.where(1 + sorted_x * idxs > cum, 1, 0), axis=1, keepdims=True)
    threshold = (jnp.take_along_axis(cum, k - 1, axis=1) - 1) / k
    
    return jnp.maximum(x - threshold, 0)