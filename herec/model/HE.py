import jax
from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

class HE(nn.Module):

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
        self.connectionMatrix += (self.param(f'connectionMatrix_1', lambda rng: jax.random.normal(rng, (self.objNum, self.clusterNums[0]), jnp.float32)),)
        for level in range(2, self.depth+1):
            self.connectionMatrix += (self.param(f'connectionMatrix_{level}', lambda rng: jax.random.normal(rng, (self.clusterNums[level-2], self.clusterNums[level-1]), jnp.float32)),)

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

        return nn.softmax(self.connectionMatrix[1][ids]) @ self.getEmbedByLevel(1)

    def getConnectionMatrix( self, level: int ):

        """
            func:
                Extract the connection matrix expressing the weights that connect the nodes at level {level}-1 to the nodes at level {level}
            args:
                level: int
                    Parent node level in the hierarchy
            returns:
                connectionMatrix: jnp.ndarray
                    Connection matrix, where rows and columns correspond to nodes at level {level}-1 and at level {level}, respectively
            Note:
                The output matrix is normalized by softmax row by row to imply a stochastic transition matrix
        """

        return nn.softmax(self.connectionMatrix[level])

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

        if (self.depth == level): return self.rootMatrix

        return jnp.linalg.multi_dot(
            [self.getConnectionMatrix(l) for l in range(level+1, self.depth+1)] + [self.rootMatrix]
        )