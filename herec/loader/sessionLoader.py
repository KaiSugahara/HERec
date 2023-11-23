import jax
import numpy as np
from tqdm import trange
import itertools

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
        LAST_FLAG = [[] for _ in range(self.batch_size)]

        # 割当
        for i in range(self.batch_size):
            tmp_session_list = session_list[i::self.batch_size]
            INPUT[i] = list(itertools.chain.from_iterable( [session[:-1] for session in tmp_session_list] ))
            OUTPUT[i] = list(itertools.chain.from_iterable( [session[1:] for session in tmp_session_list] ))
            IDS[i] = list(itertools.chain.from_iterable( [[idx]*(len(session)-1) for idx, session in enumerate(tmp_session_list)] ))

        # 余りの処理
        if self.fill_dummies:
            # 先頭をダミーIDで埋める
            batch_num = max(map(len, INPUT))
            INPUT = [[0] * (batch_num - len(row)) + row for row in INPUT]
            INPUT = np.array(INPUT)
            OUTPUT = [[0] * (batch_num - len(row)) + row for row in OUTPUT]
            OUTPUT = np.array(OUTPUT)
            IDS = [[-1] * (batch_num - len(row)) + row for row in IDS]
            IDS = np.array(IDS)
        else:
            batch_num = min(map(len, INPUT))
            INPUT = np.array([row[:batch_num] for row in INPUT])
            OUTPUT = np.array([row[:batch_num] for row in OUTPUT])
            IDS = np.array([row[:batch_num] for row in IDS])
        
        # マスク
        MASK = np.hstack([np.zeros((self.batch_size, 1), dtype=int), (IDS[:, :-1] == IDS[:, 1:]).astype(int)])
        
        # Transfer to GPU
        INPUT = jax.device_put(INPUT)
        OUTPUT = jax.device_put(OUTPUT)
        MASK = jax.device_put(MASK)

        # 保持
        self.INPUT = INPUT
        self.OUTPUT = OUTPUT
        self.MASK = MASK

        return self

    def __init__(self, key, df_DATA, batch_size, fill_dummies=False):
        
        self.fill_dummies = fill_dummies
        
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