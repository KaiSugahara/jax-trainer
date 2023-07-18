import jax

class dataLoader:

    def __init__(self, key, X, Y, batch_size):
        
        # 保持
        self.X = X
        self.Y = Y
        self.batch_size = batch_size

        # 確認
        if X.shape[0] != Y.shape[0]:
            raise Except("入力数とラベル数が一致していません。")
        
        # データ数
        self.data_size = X.shape[0]
        
        # バッチ数
        self.batch_num = self.data_size // self.batch_size
        
        # インデックスを生成
        self.shuffled_indices = jax.random.permutation(key, self.data_size)

    def __iter__(self):
        
        # バッチindexを初期化
        self.batch_idx = 0
        
        return self

    def __next__(self):
        
        if self.batch_idx > self.batch_num:
            
            # 終了
            raise StopIteration()
            
        else:
            
            # データindexをシャッフル
            indices = self.shuffled_indices[ self.batch_idx*self.batch_size : (self.batch_idx+1)*self.batch_size ]
            
            # バッチindexを更新
            self.batch_idx += 1
            
            return self.X[indices], self.Y[indices]