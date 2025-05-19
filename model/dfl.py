import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import numpy as np
from scipy.optimize import linprog

class DFL(torch.nn.Module):

    def __init__(self, A, b, verbose = True):
        super().__init__()
        self.A = torch.tensor(A).float()
        self.b = torch.tensor(b).float()
        self.d = self.A.shape[1]

        # differentiable optimization layer
        z       = cp.Variable(self.d)
        Y_param = cp.Parameter(self.d)
        A_param = cp.Parameter(self.A.shape)
        b_param = cp.Parameter(self.b.shape)
        objective   = cp.Maximize(Y_param @ z)
        constraints = [A_param @ z <= b_param]
        problem     = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        self.layer  = CvxpyLayer(problem, parameters=[Y_param, A_param, b_param], variables=[z])

        # self.Y_hat = torch.nn.Parameter(torch.zeros(self.d).float() - 0.01)
        self.Y_hat = torch.nn.Parameter(torch.randn(self.d).float() * 0.01)

        self.verbose = verbose


    def loss(self, Y):
        '''
        Args:
        - Y:    [ batch_size, dim ] torch
        Returns:
        - loss: [ batch_size ] torch
        '''
        # since it is a maximization problem, take the negative here
        z_hat,  = self.layer(self.Y_hat, self.A, self.b)    # [ dim ] torch
        obj     = Y @ z_hat.reshape(-1, 1)  # [ batch_size, 1 ] torch
        loss    = - obj # [ batch_size, 1 ]
        return loss
    
    def fit(self, Y, lr, niter, patience):
        Y = torch.tensor(Y).float()
        opt = torch.optim.Adam(params=self.parameters(), lr=lr)

        losses, Y_hats = [], []
        for i in range(niter):
            opt.zero_grad()
            loss = self.loss(Y)
            loss = loss.mean()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            Y_hats.append(self.Y_hat.clone().detach().numpy())

            if i % (niter // 10) == 0 and self.verbose:
                print(f'Iter: {i}, \t Avg Obj Val: {- losses[-1]}')

            if len(losses) - np.argmin(losses) > patience and self.verbose:
                print('Early stopping...')
                break
        return Y_hats

    def get_action(self):
        bounds  = [(None, None)] * len(self.Y_hat)
        result  = linprog(-self.Y_hat.detach().numpy(), A_ub = self.A.detach().numpy(), b_ub = self.b.detach().numpy(), bounds=bounds)
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
        'niter':    1000,
        'patience': 1000
    }

    dfl = DFL(**kwds)
    Y_hats = dfl.fit(**fit_kwds)
    dfl.get_action()

    import matplotlib.pyplot as plt

    plt.scatter(*np.array(Y_hats).T)
    plt.show()