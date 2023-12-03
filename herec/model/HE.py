import jax
from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

class HE(nn.Module):

    objNum: int
    clusterNums: Sequence[int]
    embedDim: int
    k: int

    def setup( self ):

        # Hierarchy Depth
        self.depth = len(self.clusterNums)

        # Root-node Embeddings
        self.rootMatrix = self.param(f'rootMatrix', lambda rng: jax.random.normal(rng, (self.clusterNums[-1], self.embedDim), jnp.float32))

        # Connection Matrices
        self.connectionMatrix = (None,)
        self.connectionMatrix += (self.param(f'connectionMatrix_1', lambda rng: jax.random.normal(rng, (self.objNum, self.clusterNums[0]), jnp.float32)),)
        for level in range(2, self.depth+1):
            self.connectionMatrix += (self.param(f'connectionMatrix_{level}', lambda rng: jax.random.normal(rng, (self.clusterNums[level-2], self.clusterNums[level-1]), jnp.float32)),)
            
    def topkWeightedMean(self, connectionVector, parentMatrix, k):
        
        """
            func:
                Weighted averaging of parentMatrix rows based on connectionVector weights
            args:
                connectionVector: jnp.array[k]
                    weights
                parentMatrix: jnp.array[k, d]
            return:
                embedMatrix: jnp.array[d]
        """

        topk_indices = connectionVector.argsort()[-k:]
        return nn.softmax(connectionVector[topk_indices]) @ parentMatrix[topk_indices]

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

        return jax.vmap(self.topkWeightedMean, in_axes=(0, None, None), out_axes=0)(self.connectionMatrix[1][ids], self.getEmbedByLevel(1), self.k)

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
            return jax.vmap(self.topkWeightedMean, in_axes=(0, None, None), out_axes=0)(self.connectionMatrix[level+1], self.getEmbedByLevel(level+1), self.k)