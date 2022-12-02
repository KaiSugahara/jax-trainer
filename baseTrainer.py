import numpy as np

import jax
import jax.numpy as jnp

from flax.training import train_state
import optax

from functools import partial
from tqdm import tqdm

class baseTrainer:

    def plot_loss_history(self):

        """
            損失履歴をプロット
        """

        import pandas as pd
        import plotly.express as px

        # DataFrameに変換する
        loss_history = pd.DataFrame(self.loss_history).T
        loss_history = loss_history.loc[:, ~loss_history.isnull().any()]
        loss_history.index.name = "epoch"
        loss_history.columns.name = "LABEL"

        # Plotly
        fig = px.line(loss_history)
        fig.update_yaxes(title_text="loss")

        return fig

    def calc_test_loss(self, epoch_idx, key, state, variables, X_TEST, Y_TEST):

        """
            func: 汎化誤差を計算
            args:
                - epoch_idx: エポック番号
                - key: PRNGkey
                - state: パラメータ状態
                - variables: 状態変数（carryなど）
                - X_TRAIN: テスト入力データ
                - Y_TRAIN: テスト正解データ
            returns: なし
        """

        # データローダ（ミニバッチ）
        loader = self.dataLoader(key, X_TEST, Y_TEST, batch_size=self.batch_size)

        # 損失格納用
        loss_list = []

        # ミニバッチ学習
        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_idx+1}/{self.epoch_nums}]") as pbar:
            for X, Y in pbar:
                _, loss, variables = self.train_batch(state, variables, X, Y) # stateは更新させない！
                loss_list.append(loss)
                pbar.set_postfix({"TEST_LOSS（TMP）": loss})

        self.loss_history[epoch_idx+1]["TEST_LOSS（BATCH_WISE_AVERAGE）"] = np.mean(loss_list)

    @partial(jax.jit, static_argnums=0)
    def train_batch(self, state, variables, X, Y):

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
        """

        # 勾配を計算
        (loss, variables), grads = jax.value_and_grad(self.loss_function, has_aux=True)(state.params, variables, X, Y)

        # 更新
        state = state.apply_gradients(grads=grads)
        return state, loss, variables
    
    
    # @partial(jax.jit, static_argnums=0)
    def train_epoch(self, epoch_idx, key, state, variables, X_TRAIN, Y_TRAIN):

        """
            func: エポック単位の学習
            args:
                - epoch_idx: エポック番号
                - key: PRNGkey
                - state: パラメータ状態
                - variables: 状態変数（carryなど）
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
            returns:
                - state: パラメータ状態
                - variables: 状態変数
        """

        # データローダ（ミニバッチ）
        loader = self.dataLoader(key, X_TRAIN, Y_TRAIN, batch_size=self.batch_size)

        # 損失格納用
        loss_list = []

        # ミニバッチ学習
        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_idx+1}/{self.epoch_nums}]") as pbar:
            for X, Y in pbar:
                state, loss, variables = self.train_batch(state, variables, X, Y)
                loss_list.append(loss)
                pbar.set_postfix({"TRAIN_LOSS（TMP）": loss})

        # 平均訓練損失を保存
        self.loss_history[epoch_idx+1] = {"TRAIN_LOSS（BATCH_WISE_AVERAGE）": np.mean(loss_list)}

        return state, variables
    

    def fit(self, X_TRAIN, Y_TRAIN, X_TEST=None, Y_TEST=None, params=None):

        """
            func:モデルの学習
            args:
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
                - X_TEST: テスト入力データ（任意; 汎化誤差を確認したいとき）
                - Y_TEST: テスト正解データ（任意; 汎化誤差を確認したいとき）
                - params: モデルパラメータの初期値（任意; 事前学習を行いたいとき）
            returns:
                - params: モデルパラメータ（学習済）
                - variables: 状態変数（最後）
        """

        # PRNG keyを生成
        key = jax.random.PRNGKey(self.seed)

        # 事前学習なし → パラメータの初期化
        if params is None:
            key, subkey = jax.random.split(key)
            X, Y = next(iter(self.dataLoader(subkey, X_TRAIN, Y_TRAIN, batch_size=self.batch_size))) # データローダからミニバッチを1つ取り出す
            key, subkey = jax.random.split(key)
            variables, params = self.model.init(subkey, X).pop("params")

        # Optimizer
        tx = optax.adam(self.learning_rate)

        # モデルパラメータの状態
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        # 損失のリストを作成
        self.loss_history = {}

        # 学習
        for epoch_idx in range(self.epoch_nums):

            # パラメータと状態変数の更新
            key, subkey = jax.random.split(key)
            state, variables = self.train_epoch(epoch_idx, subkey, state, variables, X_TRAIN, Y_TRAIN)

            # 平均汎化損失を計算
            self.calc_test_loss(epoch_idx, subkey, state, variables, X_TEST, Y_TEST)

        return state.params, variables

    def __init__(self, model, dataLoader, epoch_nums=128, batch_size=512, learning_rate=0.001, seed=0, **hyper_params):

        """
            初期化
        """
        
        # Flaxモデルの格納
        self.model = model

        # データローダの格納
        self.dataLoader = dataLoader

        # パラメータの格納
        self.epoch_nums = epoch_nums
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.hyper_params = hyper_params