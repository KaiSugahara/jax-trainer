import jax
import jax.numpy as jnp

from flax.training import train_state
import optax

from functools import partial
from tqdm import trange

class baseTrainer:

    def calc_current_loss(self, i, params, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST):

        """
            現在の損失を計算・保存
        """

        # 訓練誤差
        train_loss = self.loss_function(params, X_TRAIN, Y_TRAIN)

        # 汎化誤差
        test_loss = self.loss_function(params, X_TEST, Y_TEST) if (X_TEST is not None) and (Y_TEST is not None) else None

        # 保存
        self.loss_history.append({
            "epoch": i,
            "train": train_loss,
            "test": test_loss,
        })

        # tqdmの表示を更新
        if self.pbar:
            self.pbar.set_description(f"[Epoch {str(i).zfill(self.epoch_nums_digits)}/{self.epoch_nums}]")
            self.pbar.set_postfix({"TRAIN_LOSS": train_loss, "TEST_LOSS": test_loss})


    def plot_loss_history(self, hide_init_loss=False):

        """
            損失履歴をプロット
        """

        import pandas as pd
        import plotly.express as px

        # DataFrameに変換する
        loss_history = pd.DataFrame(self.loss_history)
        loss_history = loss_history.set_index("epoch")
        loss_history = loss_history.loc[:, ~loss_history.isnull().any()]
        loss_history = loss_history.iloc[1:] if hide_init_loss else loss_history

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
        grads = jax.grad(self.loss_function)(state.params, X, Y)

        # 更新
        state = state.apply_gradients(grads=grads)
        return state
    
    
    # @partial(jax.jit, static_argnums=0)
    def train_epoch(self, key, state, X_TRAIN, Y_TRAIN):

        """
            エポック単位の学習
        """

        # 入力データのインデックスをシャッフル
        key, subkey = jax.random.split(key)
        train_indices = jax.random.permutation(subkey, X_TRAIN.shape[0])

        for batch_idx in range(self.batch_nums):

            # ミニバッチを抽出
            target_indices = jax.lax.dynamic_slice_in_dim(train_indices, (batch_idx*self.batch_size), self.batch_size)
            X, Y = X_TRAIN[target_indices], Y_TRAIN[target_indices]

            # ミニバッチで更新
            state = self.train_batch(state, X, Y)

        return key, state
    

    def fit(self, X_TRAIN, Y_TRAIN, X_TEST=None, Y_TEST=None, epoch_nums=128, batch_size=512, learning_rate=0.001, seed=0):

        """
            モデルの学習
        """

        # ハイパーパラメータの初期化
        self.epoch_nums = epoch_nums
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        # バッチ数（余りは切り捨て）
        self.batch_nums = X_TRAIN.shape[0] // self.batch_size

        # 桁をカウント
        self.epoch_nums_digits = len(str(self.epoch_nums))

        # PRNG keyを生成
        key = jax.random.PRNGKey(self.seed)

        # モデルパラメータの初期化
        key, subkey = jax.random.split(key)
        params = self.model.init(subkey, X_TRAIN[:1, :])["params"]

        # Optimizer
        tx = optax.adam(self.learning_rate)

        # モデルパラメータの状態
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        # 損失のリストを作成
        self.loss_history = []

        # 学習
        with trange(self.epoch_nums) as self.pbar:
            
            # 損失を計算
            self.calc_current_loss(0, state.params, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

            for epoch_idx in self.pbar:
                # モデルパラメータの更新
                key, state = self.train_epoch(key, state, X_TRAIN, Y_TRAIN)
                # 損失を計算
                self.calc_current_loss(epoch_idx+1, state.params, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

        return state
    

    def __init__(self, model):

        """
            初期化
        """
        
        self.model = model