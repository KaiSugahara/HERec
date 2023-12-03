from flax import linen as nn

def topkWeightedMean(connectionVector, parentMatrix, k):
        
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