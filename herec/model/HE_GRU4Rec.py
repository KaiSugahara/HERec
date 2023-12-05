import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

from .HE import HE

class HE_GRU4Rec(nn.Module):
    
    item_num: int
    itemClusterNums: list
    embedDim: int
    GRU_LAYER_SIZES: Sequence[int]
    FF_LAYER_SIZES: Sequence[int]
    temperature: float
    
    def setup(self):

        self.itemEmbedder = HE(
            objNum = self.item_num,
            clusterNums = self.itemClusterNums,
            embedDim = self.embedDim,
            temperature = self.temperature,
        )
    
    @nn.compact
    def __call__(self, INPUT):

        """
            model: 指定アイテムのレーティングを予測するGRU4Rec
            args:
                INPUT: 入力データのタプル
                    X: 
                        - 入力アイテム
                        - 長さ {batch_size} の jnp.array
                    carry_mask:
                        - carryを初期化するかどうか
                        - 0=リセット, 1=引き継ぎ
                        - batch_size × 1 のjnp.array
            returns:
                X:
                    - 予測結果
        """

        X, carry_mask = INPUT
        
        # Embed Layer
        X = self.itemEmbedder.getEmbed(X)
        
        # GRU Layer(s)
        for i, size in enumerate(self.GRU_LAYER_SIZES):
            carry = self.variable('carry', str(i), jnp.zeros, (X.shape[0], size))
            carry.value = carry.value[:X.shape[0]] * carry_mask
            carry.value, X = nn.GRUCell(features=size)(carry.value, X)
            
        # Feedforward Layer(s)
        for features in self.FF_LAYER_SIZES:
            X = nn.Dense(features=features)(X)
            X = nn.relu(X)
            
        # Output:
        X = nn.Dense(features=self.item_num)(X)
        X = nn.softmax(X)
        
        return X
    
    def regularization_terms(self):
        
        return 0 + self.userEmbedder.regularization_terms() + self.itemEmbedder.regularization_terms()