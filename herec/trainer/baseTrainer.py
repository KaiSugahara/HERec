import numpy as np
from collections import defaultdict
import mlflow

import jax
import jax.numpy as jnp

from flax.training import train_state
import optax

from functools import partial
from tqdm import tqdm

from pathlib import Path
import orbax.checkpoint
from flax.training import orbax_utils

class baseTrainer:

    def get_best_epochid(self):

        """
            func: search epochid with best validation score from history
            args: None
            returns:
                epochid: int
                    epochid with best validation score
        """

        return sorted(self.loss_history.items(), key=(lambda tup: tup[1]["VALID_LOSS"]))[0][0]
    
    def get_best_params(self):

        """
            func: search model parameters with best validation score
            args: None
            returns:
                params
        """

        return self.checkpoint_manager.restore(step=self.get_best_epochid())

    def __get_key(self, is_init=False):
        
        """
            func: PRNG keyを生成
        """

        # keyを初期化
        if is_init:
            self.key = jax.random.PRNGKey(self.seed)

        # 次のkeyを生成
        self.key, subkey = jax.random.split(self.key)

        return subkey

    def __early_stopping(self, epoch_idx):

        """
            func: Early Stoppingの判定
        """

        # EarlyStoppingを実施しない場合は継続判定
        if self.es_patience == 0:
            return False

        # 検証ロスが計算されていない場合は継続判定
        if "VALID_LOSS" not in self.loss_history[epoch_idx+1].keys():
            return False
        
        # ベスト検証ロスを更新できない場合はカウント & 終了判定
        if getattr(self, "_es_best_loss", self.loss_history[0]["VALID_LOSS"]) < self.loss_history[epoch_idx+1]["VALID_LOSS"]:
            self._es_counter = getattr(self, "_es_counter", 0) + 1
            if self._es_counter == self.es_patience:
                return True
        # ベスト検証ロスを更新する場合
        else:
            self._es_counter = 0
            self._es_best_loss = self.loss_history[epoch_idx+1]["VALID_LOSS"]
            
        return False

    def score(self, params, df_DATA):

        """
            func: 入力されたx, yからロスを計算
        """

        # データローダの生成
        loader = self.dataLoader(self.__get_key(), df_DATA, batch_size=self.batch_size)
        # バッチごとのロス
        batch_loss_list = []
        # バッチごとのサイズ
        batch_size_list = []
        # 状態変数の初期化
        variables = self.variables
        # ミニバッチ単位でロスを計算
        for i, (X, Y) in enumerate(loader):
            loss, variables = self.loss_function(params, variables, X, Y)
            batch_size_list.append(X.shape[0])
            batch_loss_list.append(loss)

        # 平均値を返す
        return np.average(np.array(batch_loss_list), weights=np.array(batch_size_list))


    def __calc_current_loss(self, epoch_idx, df_TRAIN, df_VALID):

        """
            func: 現エポックのロスを計算
            args:
                - epoch_idx: エポック番号
                - df_TRAIN: 訓練入力データ
                - df_VALID: 検証入力データ
        """

        # Calc. Validation Score
        if df_VALID is not None:

            if hasattr(self, 'custom_score'):
                # Calc. Custom Score
                self.loss_history[epoch_idx+1][f"VALID_LOSS"] = (loss := self.custom_score(self.state.params, df_VALID, epoch_idx))
            else:
                # Calc. Loss as Score
                self.loss_history[epoch_idx+1][f"VALID_LOSS"] = (loss := self.score(self.state.params, df_VALID))
            
            # Save Score to MLflow
            mlflow.log_metric("VALID_LOSS", loss, step=epoch_idx+1)

        # Print
        if self.verbose > 0:
            print(f"\r[Epoch {epoch_idx+1}/{self.epoch_nums}]", end=" ")
            for key, val in self.loss_history[epoch_idx+1].items():
                print(key, val, end=" ")

        return self


    @partial(jax.jit, static_argnums=0)
    def __train_batch(self, state, variables, X, Y):

        """
            func: バッチ単位の学習
            args:
                - state: パラメータ状態
                - variables: 状態変数（carryなど）
                - X: 入力データ
                - Y: 正解データ
            returns:
                - state: パラメータ状態
                - loss: 損失
                - variables: 状態変数
            note:
                JITコンパイルしているため、stateとvariablesは引数として受け取る
        """

        # 勾配を計算
        (loss, variables), grads = jax.value_and_grad(self.loss_function, has_aux=True)(state.params, variables, X, Y)

        # 更新
        state = state.apply_gradients(grads=grads)
        return state, variables, loss


    def __train_epoch(self, epoch_idx, df_TRAIN):

        """
            func: エポック単位の学習
            args:
                - epoch_idx: エポック番号
                - df_TRAIN: 訓練入力データ
        """

        # データローダ（ミニバッチ）
        loader = self.dataLoader(self.__get_key(), df_TRAIN, batch_size=self.batch_size)

        # ミニバッチ学習
        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_idx+1}/{self.epoch_nums}]", disable=(self.verbose != 2)) as pbar:
            
            # 平均ミニバッチ損失を初期化
            self.loss_history[epoch_idx+1][f"TRAIN_LOSS(M.B.AVE.)"] = []
            # ミニバッチ学習
            for X, Y in pbar:
                # モデルパラメータ更新
                self.state, self.variables, loss = self.__train_batch(self.state, self.variables, X, Y)
                # ミニバッチのロスを表示
                pbar.set_postfix({"TRAIN_LOSS（TMP）": loss})
                # ミニバッチ損失を加算
                self.loss_history[epoch_idx+1][f"TRAIN_LOSS(M.B.AVE.)"].append(loss)
            # 平均ミニバッチ損失を計算
            self.loss_history[epoch_idx+1][f"TRAIN_LOSS(M.B.AVE.)"] = (save_loss := np.mean(self.loss_history[epoch_idx+1][f"TRAIN_LOSS(M.B.AVE.)"]))
            mlflow.log_metric("TRAIN_LOSS/REF.", save_loss, step=epoch_idx+1)    # MLFlowに保存

        return self

    def fit(self, df_TRAIN, df_VALID=None, init_params=None, init_variables=None):

        """
            func:モデルの学習
            args:
                - df_TRAIN: 訓練データ
                - df_VALID: 検証データ（任意; 汎化誤差を確認したいとき）
                - init_params: モデルパラメータの初期値（任意; 事前学習済みの場合）
                - init_variables: 状態変数の初期値（任意; 事前学習済みの場合）
        """

        # PRNG keyを初期化
        _ = self.__get_key(is_init=True)

        # パラメータの初期化（＝事前学習なし）
        if (init_params is None) or (init_variables is None):
            X, Y = next(iter(self.dataLoader(self.__get_key(), df_TRAIN, batch_size=self.batch_size))) # データローダからミニバッチを1つだけ取り出す
            self.variables, params = self.model.init(self.__get_key(), X).pop("params")
        # パラメータのセット（＝事前学習あり）
        else:
            params = init_params
            self.variables = init_variables

        # 定義：Optimizer
        tx = optax.adamw(learning_rate=self.learning_rate, weight_decay=(self.weight_decay/self.learning_rate))

        # 定義：モデルパラメータの状態
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        # 定義：CheckpointManager
        ckpt_path = Path(self.ckpt_dir) / Path(self.run.info.run_id)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            ckpt_path,
            orbax.checkpoint.PyTreeCheckpointer(),
            options=orbax.checkpoint.CheckpointManagerOptions(
                max_to_keep=(self.es_patience+1 if self.es_patience > 0 else None),
                create=True,
            )
        )

        # 損失履歴リストを初期化
        self.loss_history = defaultdict(dict)

        # 現在のロスを計算
        self.__calc_current_loss(-1, df_TRAIN, df_VALID)

        # Save Checkpoint at step=0
        ckpt = self.state.params
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})

        # 学習
        for epoch_idx in range(self.epoch_nums):

            # モデルパラメータと状態変数の更新
            self.__train_epoch(epoch_idx, df_TRAIN)

            # 現在のロスを計算
            self.__calc_current_loss(epoch_idx, df_TRAIN, df_VALID)

            # Save Checkpoint
            ckpt = self.state.params
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpoint_manager.save(epoch_idx+1, ckpt, save_kwargs={'save_args': save_args})

            # EarlyStopping判定
            if self.__early_stopping(epoch_idx): break

        return self

    def clear_cache(self):

        self.__train_batch.clear_cache()
        self.loss_function.clear_cache()

        return self


    def __init__(self, model, dataLoader, run, ckpt_dir, epoch_nums=128, batch_size=512, learning_rate=0.001, seed=0, verbose=2, weight_decay=0, es_patience=0, **other_params):

        """
            args:
                model: a model using Flax
                dataLoader: loader of dataset
                run: run of MLFlow
                ckpt_dir: save dir. of model parameter checkpoint
                epoch_nums: # of epochs
                batch_size: the size of Mini-batch
                learning_rate: learning rate of AdamW
                seed: random seed of initializer
                verbose: print of status（2: all print, 1: part print, 0: nothing）
                weight_decay: weight_decay of Adam
                other_params: model-specific hyper-parameters (options)
                es_patience: patience to judge early stopping
        """

        # Get arguments
        args = locals().copy()
        # Exclude "self"
        del args["self"]
        # Set arguments as class members
        for key, value in args.items(): setattr(self, key, value)