import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

from .baseTrainer import baseTrainer

class sessionTrainer(baseTrainer):

    def score(self, df_DATA):

        """
            func: 入力されたx, yからロスを計算
        """

        # データローダの生成
        loader = self.dataLoader(jax.random.PRNGKey(0), df_DATA, batch_size=self.batch_size)
        # バッチごとのロス
        batch_loss_list = []
        # バッチごとのサイズ
        batch_size_list = []
        # 状態変数の初期化
        variables = self.variables
        # ミニバッチ単位でロスを計算
        for i, (X, Y) in enumerate(loader):
            loss, variables = self.loss_function(self.state.params, variables, X, Y)
            batch_size_list.append(X[0].shape[0])
            batch_loss_list.append(loss)

        # 平均値を返す
        return np.average(np.array(batch_loss_list), weights=np.array(batch_size_list))

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