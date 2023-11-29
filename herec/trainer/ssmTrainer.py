import jax
import jax.numpy as jnp

import mlflow
import polars as pl
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from functools import partial

from .baseTrainer import baseTrainer

class ssmTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def __calc_top_items(self, params, user_ids):
    
        # Predict Scores for All Items
        Y = self.model.apply({"params": params}, user_ids, method=self.model.get_all_scores_by_user_ids)
        
        # Top Rows
        pred_items = (-Y).argsort(axis=1)[:, :100]

        return pred_items
    
    def custom_score(self, params, df_VALID, epoch_i):

        # Extract DATA from df_VALID
        user_ids = df_VALID["user_ids"]
        true_items = df_VALID["true_items"]
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

        # Predict Scores for Positive and Negative Items
        PRED = jnp.hstack([self.model.apply({'params': params}, X[:, (0, i)]) for i in range(1, X.shape[1])])

        # Normalize by Softmax
        PRED = jax.nn.softmax(PRED)

        # Calculate SSM Loss
        loss = - jnp.mean(jnp.log(PRED[:, 0]))

        return loss, variables

    def clear_cache(self):

        super().clear_cache()
        self.__calc_top_items.clear_cache()

        return self