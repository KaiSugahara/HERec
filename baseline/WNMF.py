import numpy as np

class WNMF():

    def __init__(self, d, lam, seed, max_iter=200):

        """
            args:
                - d: int
                    Dimension of latent vectors
                - lam: float
                    Regularization term
                - seed: int
                    random seed
                - max_iter: int, default=200
                    Maximum number of iterations before timing out.
        """

        self.d = d
        self.lam = lam
        self.seed = seed
        self.max_iter = max_iter

    def fit(self, X, W):

        """
            func: learning the latent martices
            args:
                - X: np.ndarray
                    user-by-item matrix
                - W: np.ndarray
                    weighted matrix
        """

        self.fit_transform(X, W)

        return self

    def fit_transform(self, X, W):

        """
            func: learning the latent martices
            args:
                - X: np.ndarray
                    user-by-item matrix
                - W: np.ndarray
                    weighted matrix
        """

        # Seed
        np.random.seed(self.seed)

        # Initialize 
        U = np.abs(np.random.randn(X.shape[0], self.d)) # Latent Matrix of row objects
        V = np.abs(np.random.randn(self.d, X.shape[1])) # Latent Matrix of column objects

        for i in range(self.max_iter):

            # Update U
            upper, lower = ((W * X) @ V.T), ((W * (U @ V)) @ V.T + self.lam * U)
            U = U * np.sqrt(np.divide(upper, lower, out=np.zeros_like(upper), where=(lower!=0)))
            U = np.nan_to_num(U)
            
            # Update V
            upper, lower = (U.T @ (W * X)), (U.T @ (W * (U @ V)) + self.lam * V)
            V = V * np.sqrt(np.divide(upper, lower, out=np.zeros_like(upper), where=(lower!=0)))
            V = np.nan_to_num(V)
            
            # Calc Pred Matirx
            X_pred = U @ V

        self.U, self.V = U.copy(), V.copy()

        return U, V