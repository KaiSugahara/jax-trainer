import numpy as np
from collections import defaultdict

import jax
import jax.numpy as jnp

from flax.training import train_state
import optax

from functools import partial
from tqdm import tqdm

class baseTrainer:

    def plot_loss_history(self):

        """
            func: 損失履歴をプロット
        """

        import pandas as pd
        import plotly.express as px

        # DataFrameに変換する
        loss_history = pd.DataFrame(self.loss_history).T
        loss_history.index.name = "epoch"
        loss_history.columns.name = "LABEL"

        # Plotly
        fig = px.line(loss_history)
        fig.update_yaxes(title_text="loss")

        return fig

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


    def score(self, x, y):

        """
            func: 入力されたx, yからロスを計算
        """

        # データローダの生成
        loader = self.dataLoader(self.__get_key(), x, y, batch_size=self.batch_size)
        # バッチごとのロス
        batch_loss_list = []
        # バッチごとのサイズ
        batch_size_list = []
        # 状態変数の初期化
        variables = self.variables
        # ミニバッチ単位でロスを計算
        for i, (X, Y) in enumerate(loader): 
            loss, variables = self.loss_function(self.state.params, variables, X, Y)
            batch_size_list.append(X.shape[0])
            batch_loss_list.append(loss)

        # 平均値を返す
        return np.average(np.array(batch_loss_list), weights=np.array(batch_size_list))


    def __calc_current_loss(self, epoch_idx, X_TRAIN, Y_TRAIN, X_VALID, Y_VALID):

        """
            func: 現エポックのロスを計算
            args:
                - epoch_idx: エポック番号
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
                - X_VALID: 検証入力データ
                - Y_VALID: 検証正解データ
        """

        print_objects = []

        if self.calc_fullbatch_loss:

            # 訓練ロスを計算
            self.loss_history[epoch_idx+1][f"TRAIN_LOSS"] = self.score(X_TRAIN, Y_TRAIN)

        # 検証ロスを計算
        if (X_VALID is not None) and (Y_VALID is not None):
            self.loss_history[epoch_idx+1][f"VALID_LOSS"] = self.score(X_VALID, Y_VALID)

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


    def __train_epoch(self, epoch_idx, X_TRAIN, Y_TRAIN):

        """
            func: エポック単位の学習
            args:
                - epoch_idx: エポック番号
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
        """

        # データローダ（ミニバッチ）
        loader = self.dataLoader(self.__get_key(), X_TRAIN, Y_TRAIN, batch_size=self.batch_size)

        # ミニバッチ学習
        with tqdm(loader, total=loader.batch_num, desc=f"[Epoch {epoch_idx+1}/{self.epoch_nums}]", disable=(self.verbose != 2)) as pbar:
            
            # 平均ミニバッチ損失を初期化
            self.loss_history[epoch_idx+1][f"TRAIN_LOSS（M.B. AVE.）"] = []
            # ミニバッチ学習
            for X, Y in pbar:
                # モデルパラメータ更新
                self.state, self.variables, loss = self.__train_batch(self.state, self.variables, X, Y)
                # ミニバッチのロスを表示
                pbar.set_postfix({"TRAIN_LOSS（TMP）": loss})
                # ミニバッチ損失を加算
                self.loss_history[epoch_idx+1][f"TRAIN_LOSS（M.B. AVE.）"].append(loss)
            # 平均ミニバッチ損失を計算
            self.loss_history[epoch_idx+1][f"TRAIN_LOSS（M.B. AVE.）"] = np.mean(self.loss_history[epoch_idx+1][f"TRAIN_LOSS（M.B. AVE.）"])

        return self

    def fit(self, X_TRAIN, Y_TRAIN, X_VALID=None, Y_VALID=None, init_params=None, init_variables=None):

        """
            func:モデルの学習
            args:
                - X_TRAIN: 訓練入力データ
                - Y_TRAIN: 訓練正解データ
                - X_VALID: 検証入力データ（任意; 汎化誤差を確認したいとき）
                - Y_VALID: 検証正解データ（任意; 汎化誤差を確認したいとき）
                - init_params: モデルパラメータの初期値（任意; 事前学習済みの場合）
                - init_variables: 状態変数の初期値（任意; 事前学習済みの場合）
        """

        # PRNG keyを初期化
        _ = self.__get_key(is_init=True)

        # パラメータの初期化（＝事前学習なし）
        if (init_params is None) or (init_variables is None):
            X, Y = next(iter(self.dataLoader(self.__get_key(), X_TRAIN, Y_TRAIN, batch_size=self.batch_size))) # データローダからミニバッチを1つだけ取り出す
            self.variables, params = self.model.init(self.__get_key(), X).pop("params")
        # パラメータのセット（＝事前学習あり）
        else:
            params = init_params
            self.variables = init_variables

        # 定義：Optimizer
        tx = optax.adamw(learning_rate=self.learning_rate, weight_decay=self.weight_decay)

        # 定義：モデルパラメータの状態
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

        # 損失履歴リストを初期化
        self.loss_history = defaultdict(dict)

        # 現在のロスを計算
        self.__calc_current_loss(-1, X_TRAIN, Y_TRAIN, X_VALID, Y_VALID)

        # 学習
        for epoch_idx in range(self.epoch_nums):

            # モデルパラメータと状態変数の更新
            self.__train_epoch(epoch_idx, X_TRAIN, Y_TRAIN)

            # 現在のロスを計算
            self.__calc_current_loss(epoch_idx, X_TRAIN, Y_TRAIN, X_VALID, Y_VALID)

        return self

    def get_param(self, attr_name):

        if attr_name == "params":
            return state.params if (state := getattr(self, "state", False)) else None
        else:
            return getattr(self, name, None)

    def get_params(self, deep=True):

        """
            func: ハイパーパラメータとモデルパラメータを辞書型で返す
        """

        outputs = {}
        
        # ハイパーパラメータを取得
        attribute_list = ["epoch_nums", "batch_size", "learning_rate", "seed", "verbose", "hyper_params"]
        outputs.update( {name: getattr(self, name, None) for name in attribute_list} )

        # モデルパラメータを取得
        outputs["params"] = state.params if (state := getattr(self, "state", False)) else None
        outputs["variables"] = getattr(self, "variables", None)

        return outputs

    def set_params(self, **parameters):

        """
            func: 与えられたハイパーパラメータの値をセット
        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def clear_cache(self):

        self.__train_batch.clear_cache()
        self.loss_function.clear_cache()

        return self


    def __init__(self, model, dataLoader, epoch_nums=128, batch_size=512, learning_rate=0.001, seed=0, verbose=2, weight_decay=0, calc_fullbatch_loss=False, **other_params):

        """
            args:
                model: Flaxベースのモデル
                dataLoader: データローダ
                epoch_nums: エポック数
                batch_size: ミニバッチのサイズ
                learning_rate: 学習率
                seed: ランダムシード
                verbose: 学習プロセスの進捗表示（2: すべて表示, 1: エポック毎の表示, 0: すべて非表示）
                weight_decay: weight_decay of Adam
                calc_fullbatch_loss: エポック毎にフルバッチの訓練損失を計算し直すか？
                other_params: その他のモデル特有のハイパーパラメータ, 可変長引数
        """

        # ハイパーパラメータをセット
        local_params = locals().copy() # ローカル変数を取得
        del local_params["self"] # インスタンス自身は除外
        self.set_params(**local_params)