from WNMF import WNMF
import mlflow
import numpy as np
from tqdm import tqdm, trange

class eTREE():

    def __init__(self, R, item_clusters, lbd, mu, eta, seed, maxNumIter=1000, run=None):

        """
            args:
                - R: int
                    matrix factorization rank
                - item_clusters: List[int], example=[100, 50]
                    item cluster size at each level
                - lbd: float
                - mu: float
                - eta: float
                - seed: int
        """

        self.R = R
        self.item_clusters = item_clusters
        self.lbd = lbd
        self.mu = mu
        self.eta = eta
        self.run = run
        self.seed = seed
        self.maxNumIter = maxNumIter

    def fit(self, X, W, X_VALID=None, W_VALID=None):

        """
            func: training
            args:
                - X: user-by-item matrix, np.array
                - W: weighted matrix, np.array
                - X_VALID: user-by-item matrix (Validation), np.array
                - W_VALID: weighted matrix (Validation), np.array
        """

        np.random.seed(self.seed)

        """
            Ready of HyperParams
        """
        R = self.R
        item_clusters = self.item_clusters
        lbd = self.lbd
        mu = self.mu
        eta = self.eta
        N, M = X.shape
        Mq = {1: M}
        Mq.update(dict({i+2: num for i, num in enumerate(item_clusters)}))
        Q = len(Mq)

        """
            Step1: Initialize
        """
        model = WNMF(d=R, lam=0, seed=np.random.randint(0, 100), max_iter=1)
        B = {}
        A, B[1] = model.fit_transform(X, W)
        B[1] = B[1].T

        """
            Step2
        """
        B.update({q: np.random.normal(size=(Mq[q], R)) for q in range(2, Q+1)})
        S = {q: np.eye(Mq[q+1])[np.random.randint(0, Mq[q+1], size=Mq[q])] for q in range(1, Q)}
        D = np.eye(M)
        U = np.zeros_like(A)
        V = np.zeros_like(B[1])
        Z = {
            q: (B[q].T / np.sqrt((B[q] ** 2).sum(axis=1))).T
            for q in range(1, Q)
        }

        """
            Calc Initialize Score
        """
        X_hat = A @ B[1].T @ D

        if self.run is not None:

            train_loss = np.mean((X_hat[W == 1] - X[W == 1])**2)
            mlflow.log_metric("TRAIN_LOSS", train_loss, step=0)

            if (X_VALID is not None) and (W_VALID is not None):
                valid_loss = np.mean((X_hat[W_VALID == 1] - X_VALID[W_VALID == 1])**2)
                mlflow.log_metric("VALID_LOSS", valid_loss, step=0)
                mlflow.log_metric("BEST_VALID_LOSS", valid_loss)
                self.best_valid_loss = valid_loss
        
        counter = 0

        """
            Step3-23
        """

        for iteration in trange(self.maxNumIter):

            """
                Step4-9
            """
            # Step4
            B_tilde = D @ B[1]
            rho = np.sum(B_tilde ** 2) / (N * R)
            c = lbd + rho
            J = [np.where(i == 1)[0].tolist() for i in W]
            G = {i: B_tilde[J[i]].T @ B_tilde[J[i]] + c * np.eye(R) for i in range(N)}
            L = {i: np.linalg.cholesky(G[i]) for i in range(N)}
            F = {i: B_tilde[J[i]].T @ X[i, (J[i],)].T for i in range(N)}
            
            # Step6-9
            for _ in range(0, (admmNumIter := 5)):
                
                A_tilde = np.hstack([np.linalg.inv(L[i]).T @ np.linalg.inv(L[i]) @ (F[i] + rho * (A[(i,), :].T + U[(i,), :].T)) for i in range(N)])
                A = A_tilde.T - U
                A[A < 0] = 0
                U = U + A - A_tilde.T
            
            """
                Step10
            """
            rho = np.sum(A ** 2) / (M * R)
            c = lbd + rho
            I = [np.where(j == 1)[0].tolist() for j in W.T]
            G = {j: A[I[j]].T @ A[I[j]] + c * np.eye(R) for j in range(M)}
            L = {j: np.linalg.cholesky(G[j]) for j in range(M)}
            F = {j: A[I[j]].T @ X[(I[j],), j].T for j in range(M)}
            
            k = 1
            
            for _ in range(0, (admmNumIter := 5)):
                
                B_tilde = np.hstack([np.linalg.inv(L[j]).T @ np.linalg.inv(L[j]) @ (F[j] + rho * (B[1][(j,), :].T + V[(j,), :].T)) for j in range(M)])
                B[1] = B_tilde.T - V
                B[1][B[1] < 0] = 0
                V = V + B[1] - B_tilde.T
                k = k + 1
            
            """
                Step11
            """
            Z[1] = np.vstack([B[1][j] / np.sqrt(np.sum(B[1][j]**2)) for j in range(M)])
            h = {j: (B[1][(j,), :] @ A[I[j]].T).T for j in range(M)}
            D = np.diag([np.sum(h[j] * X[I[j]][:, (j,)]) / np.sum(h[j] * h[j]) for j in range(M)])
            
            """
                Step12-22
            """
            for _ in range(treeNumIter := 5):
                for q in range(2, Q):
                    
                    # Step15
                    v = mu + eta
                    H = S[q-1].T @ S[q-1] + v * np.eye(Mq[q])
                    L = np.linalg.cholesky(H)
                    
                    # Step16
                    B[q] = np.hstack([
                        np.linalg.inv(L).T @ np.linalg.inv(L) @ ((mu * S[q-1].T @ B[q-1][:, [j]]) + (mu * S[q] @ B[q+1][:, [j]]) + (eta * Z[q][:, [j]]))
                        for j in range(R)
                    ])
                    
                    # Step17
                    S[q-1] = np.eye(S[q-1].shape[1])[[
                        np.argmin([np.sum((B[q-1][i] - B[q][j])**2) for j in range(Mq[q])])
                        for i in range(S[q-1].shape[0])
                    ]]
                    
                    # Step18
                    Z[q] = (B[q].T / np.sqrt(np.sum(B[q]**2, axis=1))).T
            
            """
                Calc Validation Score
            """
            
            X_hat = A @ B[1].T @ D

            if self.run is not None:

                train_loss = np.mean((X_hat[W == 1] - X[W == 1])**2)
                mlflow.log_metric("TRAIN_LOSS", train_loss, step=iteration+1)

                if (X_VALID is not None) and (W_VALID is not None):
                    valid_loss = np.mean((X_hat[W_VALID == 1] - X_VALID[W_VALID == 1])**2)
                    mlflow.log_metric("VALID_LOSS", valid_loss, step=iteration+1)
                    if (valid_loss >= self.best_valid_loss) or np.isnan(valid_loss):
                        counter = counter + 1
                    else:
                        counter = 0
                        mlflow.log_metric("BEST_VALID_LOSS", valid_loss)
                        self.best_valid_loss = valid_loss
                    if counter >= 10: break
            
            self.X_hat = X_hat.copy()