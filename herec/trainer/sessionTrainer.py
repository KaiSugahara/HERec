import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

from .baseTrainer import baseTrainer

class sessionTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        """
            損失関数
        """

        # 予測値
        pred, variables = self.model.apply({'params': params, **variables}, X, mutable=list(variables.keys()))

        # 交差エントロピー誤差（省略版）
        target_preds = jax.vmap(lambda row, index: row[index], in_axes=0, out_axes=0)(pred, Y) # ターゲットidの予測値のみ取り出す（他は0で消えるので無視）
        loss = - jnp.mean(jnp.log(target_preds + 1e-8))

        return loss, variables