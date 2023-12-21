import jax
import jax.numpy as jnp

import mlflow
import polars as pl
import numpy as np
from collections import defaultdict
from tqdm import trange

from functools import partial

from .baseTrainer import baseTrainer

class bprTrainer(baseTrainer):
    
    @partial(jax.jit, static_argnums=0)
    def calc_top100_indices(self, params, target_user_ids):
        scores = self.model.apply( {"params": params}, target_user_ids, method=self.model.get_all_scores_by_user_ids )
        _, top100_indices = jax.lax.top_k(scores, 100)
        return top100_indices
    
    def custom_score(self, params, df_VALID, epoch_i):

        # Extract DATA from df_VALID
        user_ids = df_VALID["user_ids"]
        true_item_ids = df_VALID["true_item_ids"]
        true_item_len = df_VALID["true_item_len"]

        # Extract Predicted Ranking
        top100_indices = jnp.vstack([
            self.calc_top100_indices(params, user_ids[i:i+1024]) for i in trange(0, user_ids.shape[0], 1024, desc=f"[Eval. {epoch_i}/{self.epochNum}]")
        ])

        # Extract Hit Flag in Predicted Ranking
        hit_flags = jax.vmap(lambda a, b: jnp.isin(a, b), in_axes=(0, 0), out_axes=(0))(top100_indices, true_item_ids).astype(int)
        hit_flags_cumsum = hit_flags.cumsum(axis=1)

        # Calc
        metrics = {}
        for k in [10, 30, 50, 100]:

            # HitRate
            metrics[f"HitRate_{k}"] = (hit_flags_cumsum[:, k-1] > 0).mean().tolist()

            # Precision
            metrics[f"Precision_{k}"] = (hit_flags_cumsum[:, k-1] / k).mean().tolist()

            # Recall
            metrics[f"Recall_{k}"] = (hit_flags_cumsum[:, k-1] / true_item_len).mean().tolist()

            # MRR
            metrics[f"MRR_{k}"] = jnp.mean( 1 / (hit_flags[:, :k].argmax(axis=1) + 1) * (hit_flags_cumsum[:, k-1] > 0) ).tolist()

            # Coverage
            metrics[f"Coverage_{k}"] = (jnp.unique( top100_indices[:, :k] ).shape[0] / self.model.item_num)

            # nDCG
            idcg_tmp = 1 / jnp.log2(jnp.arange(2, k+2))
            dcg = jnp.where( hit_flags[:, :k], idcg_tmp, 0 ).sum(axis=1)
            idcg = idcg_tmp.cumsum()[ jnp.minimum(true_item_len, k) - 1 ]
            metrics[f"nDCG_{k}"] = (dcg / idcg).mean().tolist()

        # Calc. Ave. Score over all Users & Save to MLflow
        mlflow.log_metrics(metrics, step=epoch_i)

        return - metrics[f"nDCG_{k}"]

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        # Predict Scores for Positive Items
        PRED_pos = self.model.apply( {'params': params, **variables}, X[:, (0, 1)], mutable=list(variables.keys()) )[0]

        # Predict Scores for Negative Items
        PRED_neg = jnp.hstack([
            self.model.apply( {'params': params, **variables}, X[:, (0, i)], mutable=list(variables.keys()) )[0]
            for i in range(2, X.shape[1])
        ])

        # Calculate BPR cost for sampling pairs
        cost = - jnp.log(jax.nn.sigmoid(PRED_pos - PRED_neg))

        # Weight Cost and Average
        if Y is not None:
            cost = cost * Y
            cost = cost.sum(axis=0) / Y.sum()
            loss = cost.mean()
        else:
            cost = cost.mean(axis=0)
            loss = cost.mean()
            
        # regularization_terms
        if hasattr( self.model, "regularization_terms" ):
            sample_user_ids = X[:, 0]
            sample_item_ids = X[:, 1:].reshape(-1)
            loss += self.model.apply({'params': params, **variables}, sample_user_ids, sample_item_ids, method=self.model.regularization_terms)

        return loss, variables

    def clear_cache(self):

        super().clear_cache()
        self.__calc_top_items.clear_cache()

        return self