import jax
import jax.numpy as jnp
import numpy as np
import mlflow
from tqdm import trange

from functools import partial
from .baseTrainer import baseTrainer

class sessionTrainer(baseTrainer):
    
    @partial(jax.jit, static_argnums=0)
    def __sequential_predict(self, X, params, variables):
        return self.model.apply({'params': params, **variables}, X, mutable=list(variables.keys()))
    
    def custom_score(self, params, df_VALID, epoch_i):

        pred_items_list = []
        true_item_list = []

        batchSize = self.batch_size
        for row_index in trange(0, df_VALID.height, batchSize, desc=f"[Eval. {epoch_i}/{self.epochNum}]"):

            # Extract MiniBatch
            df_MINIBATCH = df_VALID[row_index:row_index+batchSize]

            # Set DataLoader
            loader = self.dataLoader(jax.random.PRNGKey(0), df_MINIBATCH, df_MINIBATCH.height, fill_dummies=True)

            # Initialize
            variables = self.variables

            # Sequential Prediction
            for X, Y in loader:
                pred, variables = self.__sequential_predict(X, params, variables)

            # Extract Prediction Items and Target Item
            pred_items_list.append( (-pred).argsort(axis=1)[:, :30] )
            true_item_list.append( Y.reshape(-1, 1) )
            
        # Merge
        pred_items = jnp.vstack(pred_items_list)
        true_items = jnp.vstack(true_item_list)

        # Extract Hit Flag in Predicted Ranking
        pred_flag = jax.vmap(lambda a, b: jnp.isin(a, b), in_axes=(0, 0), out_axes=(0))(pred_items, true_items)
        pred_flag = jax.device_get(pred_flag)

        # Initialize
        metrics = {}

        # Calc.
        for k in [10, 30]:
            
            # Precision
            metrics[f"Precision_{k}"] = np.mean(pred_flag[:, :k].sum(axis=1) / k)
            
            # Recall
            metrics[f"Recall_{k}"] = np.mean(pred_flag[:, :k].sum(axis=1))
            
            # MRR
            metrics[f"MRR_{k}"] = np.mean((1 / (pred_flag[:, :k].argmax(axis=1) + 1)) * pred_flag[:, :k].any(axis=1))
            
            # nDCG
            dcg = np.sum(pred_flag[:, :k] * (1 / np.log2(np.arange(k) + 2)), axis=1)
            metrics[f"nDCG_{k}"] = np.mean(dcg)

        # Calc. Ave. Score over all Users & Save to MLflow
        mlflow.log_metrics({key: np.mean(val) for key, val in metrics.items()}, step=epoch_i)

        return - np.mean(metrics[f"nDCG_10"])

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