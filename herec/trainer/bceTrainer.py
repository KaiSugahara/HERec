import jax
import jax.numpy as jnp

import mlflow
import polars as pl
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from functools import partial

from .baseTrainer import baseTrainer

class bceTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def __calc_top_items(self, params, user_ids):
    
        # Predict Scores for All Items
        Y, _ = self.model.apply(
            {'params': params, **self.variables},
            user_ids,
            method=self.model.get_all_scores_by_user_ids,
            mutable=list(self.variables.keys())
        )
        
        # Top Rows
        pred_items = (-Y).argsort(axis=1)[:, :100]

        return pred_items
    
    def custom_score(self, params, df_VALID, epoch_i):

        # Extract DATA from df_VALID
        user_ids = df_VALID["user_ids"]
        true_items = df_VALID["true_item_ids"]
        true_item_len = df_VALID["true_item_len"]

        # Extract Predicted Ranking
        pred_items = jnp.vstack([
            self.__calc_top_items(params, sub_user_ids)
            for sub_user_ids in tqdm(jnp.array_split(user_ids, max(1, len(user_ids)//1024)), desc=f"[Eval. {epoch_i}/{self.epochNum}]")
        ])

        # Extract Hit Flag in Predicted Ranking
        pred_flag = jax.vmap(lambda a, b: jnp.isin(a, b), in_axes=(0, 0), out_axes=(0))(pred_items, true_items)
        pred_flag = jax.device_get(pred_flag)

        # Initialize
        metrics = {}

        # Calc.
        for k in [10, 30, 50, 100]:
            
            # HitRate
            metrics[f"HitRate_{k}"] = pred_flag[:, :k].max(axis=1).mean()
            
            # Precision
            metrics[f"Precision_{k}"] = np.mean(pred_flag[:, :k].sum(axis=1) / k)
            
            # Recall
            metrics[f"Recall_{k}"] = np.mean(pred_flag[:, :k].sum(axis=1) / true_item_len)
            
            # MRR
            metrics[f"MRR_{k}"] = np.mean((1 / (pred_flag[:, :k].argmax(axis=1) + 1)) * pred_flag[:, :k].any(axis=1))
            
            # nDCG
            dcg = np.sum(pred_flag[:, :k] * (1 / np.log2(np.arange(k) + 2)), axis=1)
            idcg = np.array([np.sum(1 / np.log2(np.arange(min(k, l)) + 2)) for l in true_item_len.tolist()])
            metrics[f"nDCG_{k}"] = np.mean(dcg / idcg)

        # Calc. Ave. Score over all Users & Save to MLflow
        mlflow.log_metrics({key: np.mean(val) for key, val in metrics.items()}, step=epoch_i)

        return - np.mean(metrics[f"nDCG_10"])

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        # Predict Scores for Positive Items
        PRED_pos = self.model.apply( {'params': params, **variables}, X[:, (0, 1)], mutable=list(variables.keys()) )[0]

        # Predict Scores for Negative Items
        PRED_neg = jnp.hstack([
            self.model.apply( {'params': params, **variables}, X[:, (0, i)], mutable=list(variables.keys()) )[0]
            for i in range(2, X.shape[1])
        ])

        # Calculate BCE cost for sampling pairs
        PRED_pos = jnp.log(jax.nn.sigmoid(PRED_pos))
        PRED_neg = jnp.log(1 - jax.nn.sigmoid(PRED_neg))
        cost = jnp.hstack([-PRED_pos]*PRED_neg.shape[1] + [-PRED_neg])
        
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