import jax
import jax.numpy as jnp

import mlflow
import polars as pl
import numpy as np
from collections import defaultdict

from functools import partial

from .baseTrainer import baseTrainer

class bprTrainer(baseTrainer):

    @partial(jax.jit, static_argnums=0)
    def __calc_top100_items(self, params, user_id):

        # Make Matrix for Prediction
        X = jnp.array([
            [user_id] * self.model.item_num,
            jnp.arange(self.model.item_num)
        ]).T
        
        # Predict Scores for All Items
        Y = self.model.apply({"params": params}, X)
        
        # Select Items with Top100 Scores
        Y = (-Y).argsort(axis=0)
        Y = Y[:100].reshape(-1)
        
        return Y
    
    def custom_score(self, params, df_VALID, epoch_idx):

        # Initialize
        metrics = defaultdict(list)

        from tqdm import tqdm
        # Calc. Metrics for all Users in Valid. Subset
        for user_id, true_items in tqdm(df_VALID.iter_rows()):

            # Extract Predicted Ranking
            pred_items = self.__calc_top100_items(params, user_id)

            # Extract Hit Flag in Predicted Ranking
            pred_flag = np.isin(pred_items, true_items)

            # Calc.
            for k in range(1, 31):

                # Precision
                metrics[f"Precision_{k}"].append( pred_flag[:k].sum() / k )

                # Recall
                metrics[f"Recall_{k}"].append( pred_flag[:k].sum() / len(true_items) )

                # MRR
                metrics[f"MRR_{k}"].append( (1 / (pred_flag[:k].argmax() + 1)) * pred_flag[:k].any() )

                # nDCG
                dcg = np.sum(pred_flag[:k] * (1 / np.log2(np.arange(k) + 2)))
                idcg = np.sum(1 / np.log2(np.arange(min(k, len(true_items))) + 2))
                metrics[f"nDCG_{k}"].append( dcg / idcg )

        # Calc. Ave. Score over all Users & Save to MLflow
        mlflow.log_metrics({key: np.mean(val) for key, val in metrics.items()}, step=epoch_idx+1)

        return - np.mean(metrics[f"nDCG_10"])

    @partial(jax.jit, static_argnums=0)
    def loss_function(self, params, variables, X, Y):

        # Predict Scores for Positive Items
        PRED_pos = self.model.apply({'params': params}, X[:, (0, 1)])

        # Predict Scores for Negative Items
        PRED_neg = jnp.hstack([self.model.apply({'params': params}, X[:, (0, i)]) for i in range(2, X.shape[1])])

        # Calculation BPR Loss
        loss = - jnp.mean(jnp.log(jax.nn.sigmoid(PRED_pos - PRED_neg)))

        return loss, variables