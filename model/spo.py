from tqdm import tqdm
import numpy as np
from scipy.optimize import linprog
from joblib import Parallel, delayed
from scipy.optimize import linprog
import os

class SPO:
    def __init__(self, A, b, verbose = True):
        self.A = A
        self.b = b
        # self.Y_hat = np.zeros(self.A.shape[-1])
        self.Y_hat = np.random.randn(self.A.shape[-1]) * 0.01
        self.verbose = verbose

    # def fit(self, Y, lr, niter):
    #     '''
    #     Args:
    #     - Y:    [ batch_size, dim ] np
    #     '''
    #     Y_hats = []
    #     iterator = tqdm(range(niter), desc='SPO') if self.verbose else range(niter)
    #     for _ in iterator:
    #         Z_spo = [] 
    #         for j in range(len(Y)):
    #             bounds  = [(None, None)] * len(self.Y_hat)
    #             result  = linprog(2*Y[j] - self.Y_hat, A_ub = self.A, b_ub = self.b, bounds=bounds, method='highs')
    #             z_spo   = result.x
    #             Z_spo.append(z_spo) 
    #         self.Y_hat = self.Y_hat - lr * np.mean(np.array(Z_spo), 0)
    #         Y_hats.append(self.Y_hat)
    #     return Y_hats

    # def get_action(self):
    #     bounds  = [(None, None)] * len(self.Y_hat)
    #     result  = linprog(-self.Y_hat, A_ub = self.A, b_ub = self.b, bounds=bounds)
    #     return result.x
    
    # True version

    # def fit(self, Y, lr, niter):
    #     '''
    #     Args:
    #     - Y:    [ batch_size, dim ] np
    #     '''
    #     Y_hats = []
    #     iterator = tqdm(range(niter), desc='SPO') if self.verbose else range(niter)
    #     for _ in iterator:
    #         Z_spo_1, Z_spo_2 = [], [] 
    #         for j in range(len(Y)):
    #             bounds  = [(None, None)] * len(self.Y_hat)

    #             result  = linprog(-Y[j], A_ub = self.A, b_ub = self.b, bounds=bounds, method='highs')
    #             z_spo_1   = result.x

    #             result  = linprog(2*self.Y_hat + Y[j], A_ub = self.A, b_ub = self.b, bounds=bounds, method='highs')
    #             z_spo_2   = result.x

    #             Z_spo_1.append(z_spo_1) 
    #             Z_spo_2.append(z_spo_2) 

    #         self.Y_hat = self.Y_hat - lr * np.mean(2 * (np.array(Z_spo_1) - np.array(Z_spo_2)), 0)
    #         Y_hats.append(self.Y_hat)
    #     return -np.array(Y_hats)

    @staticmethod
    def solve_lp_pair(y, Y_hat, A, b):
        bounds = [(None, None)] * len(Y_hat)
        z1 = linprog(-y, A_ub=A, b_ub=b, bounds=bounds, method='highs').x
        z2 = linprog(2 * Y_hat + y, A_ub=A, b_ub=b, bounds=bounds, method='highs').x
        return z1, z2

    def fit(self, Y, lr, niter, save_folder = None):
        Y_hats = []
        iterator = tqdm(range(niter), desc='SPO') if self.verbose else range(niter)

        if save_folder is not None and os.path.exists(save_folder + '/SPO_Yhat.npy'):
            self.Y_hat = np.load(save_folder + '/SPO_Yhat.npy')
            if self.verbose:
                print('Found saved SPO Y_hat, loading...')
            return -np.array(self.Y_hat)

        for _ in iterator:
            results = Parallel(n_jobs=-1)(delayed(self.solve_lp_pair)(y, self.Y_hat, self.A, self.b) for y in Y)
            Z_spo_1, Z_spo_2 = zip(*results)
            self.Y_hat = self.Y_hat - lr * np.mean(2 * (np.array(Z_spo_1) - np.array(Z_spo_2)), axis=0)
            Y_hats.append(self.Y_hat)

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            np.save(save_folder + '/SPO_Yhat.npy', self.Y_hat)

        return -np.array(Y_hats)

    
    def get_action(self):
        bounds  = [(None, None)] * len(self.Y_hat)
        result  = linprog(self.Y_hat, A_ub = self.A, b_ub = self.b, bounds=bounds)
        return result.x

if __name__ == '__main__':

    kwds  = {
        'A':    np.array([
            [1,    1],
            [-1,   0],
            [0,   -1]
        ]),
        'b':    np.array([1, 0, 0]),
    }

    Y = np.stack([
        -1 + np.random.randn(100),
        -1 + np.random.randn(100)
    ], 1)

    fit_kwds = {
        'Y':        Y,
        'lr':       1e+0,
        'niter':    100
    }

    spo = SPO(**kwds)
    Y_hats = spo.fit(**fit_kwds)
    spo.get_action()

    import matplotlib.pyplot as plt

    plt.scatter(*np.array(Y_hats).T)
    plt.show()