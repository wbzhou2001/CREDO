import numpy as np
from scipy.optimize import linprog

class RobustOptimization:
    '''
    Robust optimization of LP where uncertainty is polygon
    '''
    def __init__(self, A, b, C, d):
        '''
        non-negativity should be included in A and C
        A, b defines the feasible region
        C, d defines the uncertainty set
        A:  [ num_const, dim ]
        b:  [ num_const ]
        C:  [ num_uncertain_const, dim ]
        d:  [ num_uncertain_const ]
        '''
        self.A = A
        self.b = b
        self.C = C
        self.d = d

    def get_action(self):
        '''
        Get the optimization
        '''
        result = linprog(self.d, A_ub= - self.A @ self.C.T, b_ub = self.b) # minimization, nonnegativity for lambda is automatically guaranteed 
        # optimal value: results.fun
        z = - self.C.T @ result.x
        return z
    
if __name__ == '__main__':

    kwds  = {
        'A':    np.array([
            [1,    1],
            [-1,   0],
            [0,   -1]
        ]),
        'b':    np.array([1, 0, 0]),
        'C':    np.array([
            [1,     0],
            [-1,    0],
            [0,     1],
            [0,    -1]
        ]),
        'd':    np.array([1, 1, 1, 1])
    }

    ro = RobustOptimization(**kwds)
    ro.get_action()