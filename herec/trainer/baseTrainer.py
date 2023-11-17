from typing import Any
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
    
    def get_best_score(self):

        """
            func: search best validation score from history
            args: None
            returns:
                score
        """

        return min([l["VALID_LOSS"] for l in self.loss_history.values()])
    
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

    def __early_stopping(self, epoch_i):

        """
            func: Early Stoppingの判定
        """

        # EarlyStoppingを実施しない場合は継続判定
        if self.es_patience == 0:
            return False

        # 検証ロスが計算されていない場合は継続判定
        if "VALID_LOSS" not in self.loss_history[epoch_i].keys():
            return False
        
        # ベスト検証ロスを更新できない場合はカウント & 終了判定
        if getattr(self, "_es_best_loss", self.loss_history[0]["VALID_LOSS"]) <= self.loss_history[epoch_i]["VALID_LOSS"]:
            self._es_counter = getattr(self, "_es_counter", 0) + 1
            if self._es_counter == self.es_patience:
                return True
        # ベスト検証ロスを更新する場合
        else:
            self._es_counter = 0
            self._es_best_loss = self.loss_history[epoch_i]["VALID_LOSS"]
            
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
            batch_size_list.append(X.shape[0] if type(X) is not tuple else X[0].shape[0])
            batch_loss_list.append(loss)

        # 平均値を返す
        return np.average(np.array(batch_loss_list), weights=np.array(batch_size_list))

    def __save_metric(self, metric_name: str, metric_value: float, epoch_i: int):
        
        """
            Save metric score for one epoch
        """
        
        self.loss_history[epoch_i][metric_name] = metric_value
        mlflow.log_metric(metric_name, metric_value, step=epoch_i)    # MLFlowに保存
        
        return self

    def __calc_current_loss(self, epoch_i, df_TRAIN, df_VALID):

        """
            func: 現エポックのロスを計算
            args:
                - epoch_i: エポック番号
                - df_TRAIN: 訓練入力データ
                - df_VALID: 検証入力データ
        """

        # Calc. Validation Score
        if df_VALID is not None:

            if hasattr(self, 'custom_score'):
                # Calc. Custom Score
                self.__save_metric("VALID_LOSS", self.custom_score(self.state.params, df_VALID, epoch_i), epoch_i)
            else:
                # Calc. Loss as Score
                self.__save_metric("VALID_LOSS", self.score(self.state.params, df_VALID), epoch_i)

        # Print
        if self.verbose > 0:
            print(f"\r[Epoch {epoch_i}/{self.epochNum}]", end=" ")
            for key, val in self.loss_history[epoch_i].items():
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

    def __train_epoch(self, epoch_i: int, df_TRAIN: Any):

        """
            Training for one epoch
            
            Args:
                - epoch_i: Index of the epoch
                - df_TRAIN: Training subset dataset
        """

        # Create an instance of the Data Loader
        loader: Any = self.dataLoader(self.__get_key(), df_TRAIN, batch_size=self.batch_size)

        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_i}/{self.epochNum}]", disable=(self.verbose != 2)) as pbar:
            
            # Initialize list to store losses temporarily
            losses = []
            
            # Perform mini-batch learning
            for X, Y in pbar:
                # Update model parameters
                self.state, self.variables, miniBatchLoss = self.__train_batch(self.state, self.variables, X, Y)
                # Display the mini-batch loss
                pbar.set_postfix({"TRAIN_LOSS（TMP）": miniBatchLoss})
                # Store the mini-batch loss
                losses.append(miniBatchLoss)
        
        # Calculate average loss as rough loss on training subset
        self.__save_metric("TRAIN_LOSS/ROUGH", np.mean(losses), epoch_i)

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
        self.__calc_current_loss(0, df_TRAIN, df_VALID)

        # Save Checkpoint at step=0
        ckpt = self.state.params
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})

        # 学習
        for epoch_i in range(1, self.epochNum+1):

            # モデルパラメータと状態変数の更新
            self.__train_epoch(epoch_i, df_TRAIN)

            # 現在のロスを計算
            self.__calc_current_loss(epoch_i, df_TRAIN, df_VALID)

            # Save Checkpoint
            ckpt = self.state.params
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpoint_manager.save(epoch_i, ckpt, save_kwargs={'save_args': save_args})

            # EarlyStopping判定
            if self.__early_stopping(epoch_i): break

        return self

    def clear_cache(self):

        self.__train_batch.clear_cache()
        self.loss_function.clear_cache()

        return self

    def __init__(
        self,
        model: Any,
        dataLoader: Any,
        run: Any,
        ckpt_dir: str,
        epochNum: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        es_patience: int,
        seed: int = 0,
        verbose: int = 2,
        **other_params: Any,
    ):

        """
            Trainer a Flax model using specified parameters.

            Args:
                model: Flax model to be trained.
                dataLoader: DataLoader for the dataset.
                run: MLFlow run for experiment tracking.
                ckpt_dir: Directory to save model parameter checkpoints.
                epochNum: Number of epochs to train.
                batch_size: Mini-batch size.
                learning_rate: Learning rate for AdamW optimizer.
                weight_decay: Weight decay for AdamW optimizer.
                es_patience: Patience for early stopping criterion.
                seed: Random seed for initialization.
                verbose: Verbosity level (2: all prints, 1: partial prints, 0: no prints).
                other_params: Model-specific hyperparameters (optional).
        """

        # Set Arguments as Class Members
        setattr(self, "model", model)
        setattr(self, "dataLoader", dataLoader)
        setattr(self, "run", run)
        setattr(self, "ckpt_dir", ckpt_dir)
        setattr(self, "epochNum", epochNum)
        setattr(self, "batch_size", batch_size)
        setattr(self, "learning_rate", learning_rate)
        setattr(self, "weight_decay", weight_decay)
        setattr(self, "es_patience", es_patience)
        setattr(self, "seed", seed)
        setattr(self, "verbose", verbose)
        for arg_name, arg_value in other_params.items(): setattr(self, arg_name, arg_value)