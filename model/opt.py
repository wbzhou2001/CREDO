from scipy.optimize import linprog
import numpy as np

class Optimization:
    '''non-negativity should be included in A'''
    def __init__(self, Y, A, b):
        self.Y = Y
        self.A = A
        self.b = b

    def get_action(self):
        bounds = [(None, None)] * len(self.Y)
        result = linprog(-self.Y, A_ub = self.A, b_ub = self.b, bounds=bounds)
        return result.x
    
if __name__ == '__main__':

    kwds  = {
        'A':    np.array([
            [1,    1],
            [-1,   0],
            [0,   -1]
        ]),
        'b':    np.array([1, 0, 0]),
        'Y':    np.array([-0.1, 0.1])
    }

    o = Optimization(**kwds)
    o.get_action()