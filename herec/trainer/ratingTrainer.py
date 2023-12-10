import jax
import jax.numpy as jnp

from functools import partial

from .baseTrainer import baseTrainer

class ratingTrainer(baseTrainer):
    
    def custom_score(self, params, df_VALID, epoch_i):

        loader = self.dataLoader(jax.random.PRNGKey(0), df_VALID, batch_size=df_VALID.height)
        X, Y = next(iter(loader))

        # 予測値
        pred, variables = self.model.apply({'params': params, **self.variables}, X, mutable=list(self.variables.keys()))

        # MSEを計算
        loss = jnp.mean((pred - Y)**2)

        return loss

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        """
            損失関数
        """

        # 予測値
        pred, variables = self.model.apply({'params': params, **variables}, X, mutable=list(variables.keys()))

        # MSEを計算
        loss = jnp.mean((pred - Y)**2)
        
        # regularization_terms
        if hasattr( self.model, "regularization_terms" ):
            loss += self.model.apply({'params': params, **variables}, method=self.model.regularization_terms)

        return loss, variables