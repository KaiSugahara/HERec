from WNMF import WNMF
import mlflow
import numpy as np
from tqdm import tqdm, trange

class IHSR():

    def __init__(self, d, n_by_level, m_by_level, lam, seed, max_iter=1000, run=None):

        """
            args:
                - d: int
                    Dimension of latent vectors
                - n_by_level: List[int], example=[100, 50, 10]
                    user cluster size at each level
                - m_by_level: List[int], example=[100, 50, 10]
                    item cluster size at each level
                - lam: float
                    Regularization term
                - seed: int
                    random seed
                - max_iter: int, default=200
                    Maximum number of iterations before timing out.
        """

        self.d = d
        self.n_by_level = n_by_level
        self.m_by_level = m_by_level
        self.lam = lam
        self.seed = seed
        self.max_iter = max_iter
        self.run = run

    def fit(self, X, W, X_VALID=None, W_VALID=None):

        """
            func: training
            args:
                - X: user-by-item matrix, np.array
                - W: weighted matrix, np.array
                - X_VALID: user-by-item matrix (Validation), np.array
                - W_VALID: weighted matrix (Validation), np.array
        """

        # Seed
        np.random.seed(self.seed)

        # User/Item Hierarchy Depth
        p = len(self.n_by_level) + 1
        q = len(self.m_by_level) + 1

        # Initialize Latent Matrices (Step1)
        U_by_level = {}
        V_by_level = {}
        U_tilde_by_level = {}
        V_tilde_by_level = {}

        # Step2
        print("\r[Step2] ...", end="")
        model = WNMF( d=self.d, lam=self.lam, seed=np.random.randint(0, 100), max_iter=self.max_iter )
        U_tilde_by_level[1], V_tilde_by_level[1] = model.fit_transform(X=X.copy(), W=W.copy())
        print("\r[Step2] done!")

        # Step3-5
        print("\r[Step3-5] ...", end="")
        for i in range(1, p):
            model = WNMF( d=self.n_by_level[i], lam=self.lam, seed=np.random.randint(0, 100), max_iter=self.max_iter )
            U_by_level[i], U_tilde_by_level[i+1] = model.fit_transform(X=U_tilde_by_level[i].copy(), W=np.ones_like(U_tilde_by_level[i]))
        print("\r[Step3-5] done!")

        # Step6-8
        print("\r[Step6-8] ...", end="")
        for i in range(1, q):
            model = WNMF( d=self.m_by_level[i], lam=self.lam, seed=np.random.randint(0, 100), max_iter=self.max_iter )
            V_tilde_by_level[i+1], V_by_level[i] = model.fit_transform(X=V_tilde_by_level[i].copy(), W=np.ones_like(V_tilde_by_level[i]))
        print("\r[Step6-8] done!")

        # Step9
        print("\r[Step9] ...", end="")
        U_by_level[p] = U_tilde_by_level[p].copy()
        V_by_level[q] = V_tilde_by_level[q].copy()
        print("\r[Step9] done!")

        # Calc Current Loss/Score
        if self.run:
            # Prediction
            X_pred = np.linalg.multi_dot( [U_by_level[i] for i in range(1, p+1)] + [V_by_level[j] for j in range(q, 0, -1)] )
            # Loss
            loss_current = np.sum((W * (X - X_pred))**2) + self.lam * sum([np.sum(U**2) for U in U_by_level.values()] + [np.sum(V**2) for V in V_by_level.values()])
            mlflow.log_metric("TRAIN_LOSS", loss_current, step=0)
            # RMSE(VALID)
            if (X_VALID is not None) and (W_VALID is not None):
                valid_loss = np.mean((X_VALID[W_VALID == 1] - X_pred[W_VALID == 1]) ** 2)
                mlflow.log_metric("VALID_LOSS", valid_loss, step=0)
                mlflow.log_metric("BEST_VALID_LOSS", valid_loss)
                self.best_valid_loss = valid_loss
                counter = 0

        # Step10-20
        for i in trange(self.max_iter, desc="[Step10-20]"):

            # Step11-14
            B = {
                i: np.linalg.multi_dot( [U_by_level[j] for j in range(1, p+1)] + [V_by_level[j] for j in range(q, i, -1)] )
                for i in range(1, q+1)
            }
            M = {
                i: np.linalg.multi_dot( [V_by_level[j] for j in range(i-1, 0, -1)] + [np.eye(X.shape[1])]*2 )
                for i in range(1, q+1)
            }
            V_by_level = {
                i: np.nan_to_num( V_by_level[i] * np.sqrt((B[i].T @ (W * X) @ M[i].T) / (B[i].T @ (W * (B[i] @ V_by_level[i] @ M[i])) @ M[i].T + self.lam * V_by_level[i])) )
                for i in range(1, q+1)
            }
            
            # Step16-19
            A = {
                i: np.linalg.multi_dot( [np.eye(X.shape[0])]*2 + [U_by_level[j] for j in range(1, i)] )
                for i in range(1, p+1)
            }
            H = {
                i: np.linalg.multi_dot( [U_by_level[j] for j in range(i+1, p+1)] + [V_by_level[j] for j in range(q, 0, -1)] )
                for i in range(1, p+1)
            }
            U_by_level = {
                i: np.nan_to_num( U_by_level[i] * np.sqrt((A[i].T @ (W * X) @ H[i].T) / (A[i].T @ (W * (A[i] @ U_by_level[i] @ H[i])) @ H[i].T + self.lam * U_by_level[i])) )
                for i in range(1, p+1)
            }

            # Step21
            X_pred = np.linalg.multi_dot( [U_by_level[i] for i in range(1, p+1)] + [V_by_level[j] for j in range(q, 0, -1)] )

            # Calc Current Loss/Score
            if self.run:
                # Prediction
                X_pred = np.linalg.multi_dot( [U_by_level[i] for i in range(1, p+1)] + [V_by_level[j] for j in range(q, 0, -1)] )
                # Loss
                loss_current = np.sum((W * (X - X_pred))**2) + self.lam * sum([np.sum(U**2) for U in U_by_level.values()] + [np.sum(V**2) for V in V_by_level.values()])
                mlflow.log_metric("TRAIN_LOSS", loss_current, step=i+1)
                # RMSE(VALID)
                if (X_VALID is not None) and (W_VALID is not None):
                    valid_loss = np.mean((X_VALID[W_VALID == 1] - X_pred[W_VALID == 1]) ** 2)
                    mlflow.log_metric("VALID_LOSS", valid_loss, step=i+1)
                    if (valid_loss >= self.best_valid_loss):
                        counter = counter + 1
                    else:
                        counter = 0
                        mlflow.log_metric("BEST_VALID_LOSS", valid_loss)
                        self.best_valid_loss = valid_loss
                    if counter >= 10: break
            
        # Store Variables
        self.X_pred = X_pred
        self.U_by_level = U_by_level
        self.V_by_level = V_by_level

        return self