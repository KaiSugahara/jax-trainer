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

    def calc_current_loss(self, epoch_idx, key, state, variables, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):

        """
            func: 現エポックの各種損失を計算
            args:
                - epoch_idx: エポック番号
                - key: PRNGkey
                - state: パラメータ状態
                - variables: 状態変数（carryなど）
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
                - X_TEST: テスト入力データ
                - Y_TEST: テスト正解データ
            returns: なし
        """

        self.loss_history[epoch_idx+1] = {}
        print_objects = [f"\r[Epoch {epoch_idx+1}/{self.epoch_nums}]"]

        # Evaluate Train Loss
        loader = self.dataLoader(key, X_TRAIN, Y_TRAIN, batch_size=self.batch_size) # データローダの作成
        loss_list = [] # 損失格納用
        t_variables = variables # 状態変数の初期化
        for i, (X, Y) in enumerate(loader): # ミニバッチ単位で損失を計算
            print(f"\r[Epoch {epoch_idx+1}/{self.epoch_nums}]", "TRAIN", f"{i+1}/{loader.batch_num}", end="")
            loss, t_variables = self.loss_function(state.params, t_variables, X, Y)
            loss_list.append(loss) # 損失格納（バッチ単位）
        self.loss_history[epoch_idx+1]["TRAIN_LOSS"] = np.mean(loss_list) # 損失の平均を保存
        print_objects += ["TRAIN_LOSS:", self.loss_history[epoch_idx+1]["TRAIN_LOSS"]]

        # Evaluate Test Loss
        if (X_TEST is not None) and (Y_TEST is not None):
            loader = self.dataLoader(key, X_TEST, Y_TEST, batch_size=self.batch_size) # データローダの作成
            loss_list = [] # 損失格納用
            t_variables = variables # 状態変数の初期化
            for i, (X, Y) in enumerate(loader): # ミニバッチ単位で損失を計算
                print(f"\r[Epoch {epoch_idx+1}/{self.epoch_nums}]", "TEST", f"{i+1}/{loader.batch_num}", end="")
                loss, t_variables = self.loss_function(state.params, t_variables, X, Y)
                loss_list.append(loss) # 損失格納（バッチ単位）
            self.loss_history[epoch_idx+1]["TEST_LOSS"] = np.mean(loss_list) # 損失の平均を保存
            print_objects += ["TEST_LOSS:", self.loss_history[epoch_idx+1]["TEST_LOSS"]]

        # Print
        print(*print_objects)


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

        # ミニバッチ学習
        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_idx+1}/{self.epoch_nums}]") as pbar:
            for X, Y in pbar:
                state, loss, variables = self.train_batch(state, variables, X, Y)
                pbar.set_postfix({"TRAIN_LOSS（TMP）": loss})

        return state, variables
    

    def fit(self, X_TRAIN, Y_TRAIN, X_TEST=None, Y_TEST=None, init_params=None, init_variables=None):

        """
            func:モデルの学習
            args:
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
                - X_TEST: テスト入力データ（任意; 汎化誤差を確認したいとき）
                - Y_TEST: テスト正解データ（任意; 汎化誤差を確認したいとき）
                - init_params: モデルパラメータの初期値（任意; 事前学習済みの場合）
                - init_variables: 状態変数の初期値（任意; 事前学習済みの場合）
            returns:
                - params: モデルパラメータ（学習済）
                - variables: 状態変数（最後）
        """

        # PRNG keyを生成
        key = jax.random.PRNGKey(self.seed)

        # 事前学習なし → パラメータの初期化
        if (init_params is None) or (init_variables is None):
            key, subkey = jax.random.split(key)
            X, Y = next(iter(self.dataLoader(subkey, X_TRAIN, Y_TRAIN, batch_size=self.batch_size))) # データローダからミニバッチを1つ取り出す
            key, subkey = jax.random.split(key)
            variables, params = self.model.init(subkey, X).pop("params")
        params = params if (init_params is None) else init_params
        variables = variables if (init_variables is None) else init_variables

        # Optimizer
        tx = optax.adam(self.learning_rate)

        # モデルパラメータの状態
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        # 損失のリストを作成
        self.loss_history = {}

        # 初期の各種損失を計算
        key, subkey = jax.random.split(key)
        self.calc_current_loss(-1, subkey, state, variables, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

        # 学習
        for epoch_idx in range(self.epoch_nums):

            # パラメータと状態変数の更新
            key, subkey = jax.random.split(key)
            state, variables = self.train_epoch(epoch_idx, subkey, state, variables, X_TRAIN, Y_TRAIN)

            # 現エポックの各種損失を計算
            key, subkey = jax.random.split(key)
            self.calc_current_loss(epoch_idx, subkey, state, variables, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

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