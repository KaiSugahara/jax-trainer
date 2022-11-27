import numpy as np

import jax
import jax.numpy as jnp

from flax.training import train_state
import optax

from functools import partial
from tqdm import tqdm

class baseTrainer:

    def save_all_loss(self, epoch_idx, params, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):

        """
            データ全体の損失を保存
        """

        # keyを設定（シャッフルしないので何でも良い）
        key = jax.random.PRNGKey(0)

        # 訓練誤差
        X, Y = next(iter(self.dataLoader(key, X_TRAIN, Y_TRAIN, batch_size=X_TRAIN.shape[0])))
        train_loss = self.loss_function(params, X, Y)

        # 汎化誤差
        if (X_TEST is None) or (Y_TEST is None):
            test_loss = None
        else:
            X, Y = next(iter(self.dataLoader(key, X_TEST, Y_TEST, batch_size=X_TEST.shape[0])))
            test_loss = self.loss_function(params, X, Y)

        # 保存
        self.loss_history[epoch_idx+1] = {"TRAIN_LISS": train_loss, "TEST_LOSS": test_loss}


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


    @partial(jax.jit, static_argnums=0)
    def train_batch(self, state, X, Y):

        """
            バッチ単位の学習
        """

        # 勾配を計算
        loss, grads = jax.value_and_grad(self.loss_function)(state.params, X, Y)

        # 更新
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    
    # @partial(jax.jit, static_argnums=0)
    def train_epoch(self, epoch_idx, key, state, X_TRAIN, Y_TRAIN):

        """
            エポック単位の学習
        """

        # データローダ（ミニバッチ）
        loader = self.dataLoader(key, X_TRAIN, Y_TRAIN, batch_size=self.batch_size)

        # 損失格納用
        loss_list = []

        # ミニバッチ学習
        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_idx+1}/{self.epoch_nums}]") as pbar:
            for X, Y in pbar:
                state, loss = self.train_batch(state, X, Y)
                loss_list.append(loss)
                pbar.set_postfix({"TRAIN_LOSS（TMP）": loss})

        # 平均損失を保存
        if self.save_loss_type == "average":
            self.loss_history[epoch_idx+1] = {"TRAIN_AVE_LOSS": np.mean(loss_list)}

        return state
    

    def fit(self, X_TRAIN, Y_TRAIN, X_TEST=None, Y_TEST=None):

        """
            モデルの学習
        """

        # PRNG keyを生成
        key = jax.random.PRNGKey(self.seed)

        # モデルパラメータの初期化
        key, subkey = jax.random.split(key)
        X, Y = next(iter(self.dataLoader(subkey, X_TRAIN, Y_TRAIN, batch_size=1))) # データローダからミニバッチを1つ取り出す
        key, subkey = jax.random.split(key)
        params = self.model.init(subkey, X)["params"]

        # Optimizer
        tx = optax.adam(self.learning_rate)

        # モデルパラメータの状態
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        # 損失のリストを作成
        self.loss_history = {}

        # 学習
        for epoch_idx in range(self.epoch_nums):

            # モデルパラメータの更新
            key, subkey = jax.random.split(key)
            state = self.train_epoch(epoch_idx, subkey, state, X_TRAIN, Y_TRAIN)

            # エポック単位で損失を保存
            if self.save_loss_type == "all":
                self.save_all_loss(epoch_idx, state.params, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

        return state
    

    def __init__(self, model, dataLoader, epoch_nums=128, batch_size=512, learning_rate=0.001, seed=0, save_loss_type="all", **hyper_params):

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
        self.save_loss_type = save_loss_type
        self.hyper_params = hyper_params