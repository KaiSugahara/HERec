import jax
import math
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

class sessionLoader:

    def make_dataset(self, session_list):
    
        """
            func: 与えられたセッションのリストからGRU4Recの学習用データ行列を生成（詳細は論文参照）
            args:
                session_list: セッション（リスト）のリスト
            returns:
                INPUT: 入力用の行列
                OUTPUT: 出力用の行列
                IDS: INPUTに対応するセッションIDの行列（マスク用に使用を想定）
        """

        # 初期化
        INPUT = [[] for _ in range(self.batch_size)]
        OUTPUT = [[] for _ in range(self.batch_size)]
        IDS = [[] for _ in range(self.batch_size)]
        INPUT_SIZE = [0 for _ in range(self.batch_size)]

        # 割当
        for session_id, session in tqdm(enumerate(session_list)):
            sorted_index = np.argmin(INPUT_SIZE)
            INPUT[sorted_index] += session[:-1]
            OUTPUT[sorted_index] += session[1:]
            IDS[sorted_index] += [session_id for _ in range(len(session)-1)]
            INPUT_SIZE[sorted_index] += len(session) - 1

        # 末尾+1はダミーIDで埋める
        # batch_num = max(map(len, INPUT)) + 1
        batch_num = max(map(len, INPUT))
        INPUT = [[0] * (batch_num - len(row)) + row for row in INPUT]
        OUTPUT = [[0] * (batch_num - len(row)) + row for row in OUTPUT]
        IDS = [[session_id+1] * (batch_num - len(row)) + row for row in IDS]

        # Numpyに変換
        self.INPUT = jnp.array(INPUT)
        self.OUTPUT = jnp.array(OUTPUT)
        self.IDS = jnp.array(IDS)
        self.MASK = jnp.hstack([jnp.zeros((self.batch_size, 1), dtype=int), (self.IDS[:, :-1] == self.IDS[:, 1:]).astype(int)])
        self.LAST_FLAG = (self.IDS != jnp.hstack([self.IDS[:, 1:], jnp.array([[self.IDS.max()]] * self.batch_size)]))

        return self

    def __init__(self, key, df_DATA, batch_size):
        
        # Set batch_size
        self.batch_size = batch_size

        # Extract GRU4Rec Format Dataset
        self.make_dataset(
            session_list = df_DATA.sample(fraction=1, shuffle=True, seed=key.tolist()[0]).get_column("item_list").to_list()
        )
        
        # Calc Batch #
        self.batch_num = self.INPUT.shape[1]

    def __iter__(self):
        
        # Initialize batch-index as 0
        self.batch_idx = 0
        
        return self

    def __next__(self):
        
        if self.batch_idx == self.batch_num:
            
            # Stop
            raise StopIteration()
            
        else:

            # Extract Input IDs
            X = self.INPUT[:, self.batch_idx]

            # Extract Labels
            Y = self.OUTPUT[:, self.batch_idx]

            # MASK for Carry Reset (reset: 0, no-reset: 1)
            carry_mask = self.MASK[:, (self.batch_idx,)]
            
            # Update batch-index
            self.batch_idx += 1
            
            return (X, carry_mask), Y