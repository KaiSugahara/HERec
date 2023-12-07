import jax
from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

class DHE(nn.Module):

    objNum: int
    clusterNums: Sequence[int]
    embedDim: int
    temperature: float

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
            
        # Denses for Connection Matrices
        self.Denses = [None] + [[nn.Dense(features=self.clusterNums[level-1]) for n in range(5)] for level in range(1, self.depth+1)]
            
    def __getConnectionMatrix( self, level, ids=None ):
        
        connectionMatrix = self.connectionMatrix[level] if ids is None else self.connectionMatrix[level][ids]

        for Dense in self.Denses[level]:
            connectionMatrix = nn.relu(Dense(connectionMatrix))
            
        return nn.softmax(connectionMatrix / self.temperature)

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

        return self.__getConnectionMatrix(1, ids) @ self.getEmbedByLevel(1)

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
            return self.__getConnectionMatrix(level+1) @ self.getEmbedByLevel(level+1)

    def regularization_terms(self):
        
        loss = 0
        
        return loss