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
        radius = self.param(f'radius', lambda rng: jnp.ones(1))
        self.rootMatrix = radius * self.variable('rootMatrix', 'embedding', lambda: self.generate_root_matrix(jax.random.PRNGKey(0), self.clusterNums[-1], self.embedDim)).value

        # Connection Matrices
        self.connectionMatrix = (None,)
        for level in range(1, self.depth+1):
            row_num = self.clusterNums[level-2] if level > 1 else self.objNum
            col_num = self.clusterNums[level-1]
            self.connectionMatrix += (self.param(f'connectionMatrix_{level}', lambda rng: jax.random.normal(rng, (row_num, col_num), jnp.float32)),)
            
    def generate_root_matrix( self, key, rootObjNum, embedDim ):
        
        rootEmbed = jax.random.normal(key, (rootObjNum, embedDim))
        rootEmbed = rootEmbed / jnp.linalg.norm(rootEmbed, axis=1, keepdims=True)
        
        return rootEmbed

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
            return nn.softmax(self.connectionMatrix[level+1]) @ self.getEmbedByLevel(level+1)