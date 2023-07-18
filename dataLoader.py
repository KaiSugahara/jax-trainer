import jax
import jax.numpy as jnp
import math

class dataLoader:

    def __init__(self, key, X, Y, batch_size):
        
        # 保持
        self.batch_size = batch_size

        # 確認
        if X.shape[0] != Y.shape[0]:
            raise Except("入力数とラベル数が一致していません。")
        
        # データ数
        self.data_size = X.shape[0]
        
        # バッチ数
        self.batch_num = math.ceil(self.data_size / self.batch_size)
        
        # インデックスをシャッフル
        self.shuffled_indices = jax.random.permutation(key, self.data_size)

        # データの行をシャッフル
        self.X = X[self.shuffled_indices].copy()
        self.Y = Y[self.shuffled_indices].copy()

    def __iter__(self):
        
        # バッチindexを初期化
        self.batch_idx = 0
        
        return self

    def __next__(self):
        
        if self.batch_idx == self.batch_num:
            
            # 終了
            raise StopIteration()
            
        else:
            
            # ミニバッチ{batch_idx}を抽出
            start_index = self.batch_size * self.batch_idx
            slice_size = min( self.batch_size, (self.data_size - start_index) )
            X = jax.lax.dynamic_slice_in_dim(self.X, start_index, slice_size)
            Y = jax.lax.dynamic_slice_in_dim(self.Y, start_index, slice_size)
            
            # バッチindexを更新
            self.batch_idx += 1
            
            return X, Y