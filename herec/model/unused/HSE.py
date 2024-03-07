import jax
from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

class HSE(nn.Module):

    objNum: int
    clusterNums: Sequence[int]
    embedDim: int

    def setup( self ):

        # Hierarchy Depth
        self.depth = len(self.clusterNums)

        # Root-node Embeddings
        self.rootMatrix = self.param(f'rootMatrix', lambda rng: jax.random.normal(rng, (self.clusterNums[-1], self.embedDim), jnp.float32))

        # Connection Matrices
        self.connectionMatrix = (None,)
        for level in range(1, self.depth+1):
            row_num = self.clusterNums[level-2] if level > 1 else self.objNum
            col_num = self.clusterNums[level-1]
            self.connectionMatrix += (self.param(f'connectionMatrix_{level}', lambda rng: jax.random.normal(rng, (row_num, col_num), jnp.float32)),)
    
    def sparsemax( self, x ):

        idxs = (jnp.arange(x.shape[1]) + 1).reshape(1, -1)
        sorted_x = jnp.flip(jax.lax.sort(x, dimension=1), axis=1)
        cum = jnp.cumsum(sorted_x, axis=1)
        k = jnp.sum(jnp.where(1 + sorted_x * idxs > cum, 1, 0), axis=1, keepdims=True)
        threshold = (jnp.take_along_axis(cum, k - 1, axis=1) - 1) / k
        
        return jnp.maximum(x - threshold, 0)

    def getEmbed( self, ids: list ):

        """
            func:
                Extract embeddings of given objects (leaf nodes)
            args:
                ids: list[int]
                    Target IDs from which to extract the embedding
            returns:
                embedMatrix: jnp.ndarray
                    Embedding matrix with rows corresponding to target object embeddings
        """

        return self.sparsemax(self.connectionMatrix[1][ids]) @ self.getEmbedByLevel(1)

    def getEmbedByLevel( self, level: int ):

        """
            func:
                Extract embeddings of nodes at given level
            args:
                level: int
                    Target node level in the hierarchy
            returns:
                embedMatrix: jnp.ndarray
                    Embedding matrix with rows corresponding to target node embeddings
        """

        if (self.depth == level):
            return self.rootMatrix

        else:
            return self.sparsemax(self.connectionMatrix[level+1]) @ self.getEmbedByLevel(level+1)