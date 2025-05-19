import numpy as np
from scipy.stats import norm

class GaussianMixture:
    
    def __init__(self, mus, sigmas, ws = None):
        self.mus    = mus       # [ ncomps ]
        self.sigmas = sigmas    # [ ncomps ]
        self.ncomp  = len(mus)
        self.ws     = ws if ws is not None else np.ones(self.ncomp) / self.ncomp  # [ ncomps ] 

    def pdf(self, xx):
        ''' [ batch_size ] -> [ batch_size ] '''
        vals = []
        for i in range(self.ncomp):
            vals.append(self.ws[i] * norm.pdf(xx, loc = self.mus[i], scale = self.sigmas[i]))
        a = np.stack(vals, 0).sum(0) # [ batch_size ]
        return a

    def sample(self, n):
        '''note that the output order is not random'''
        index   = np.random.choice(np.arange(self.ncomp), size=n, p=self.ws) # [ n ]
        unique  = np.arange(self.ncomp) # [ ncomp ]
        mask    = index[None, :] == unique[:, None]
        repeats = mask.sum(1)   # [ ncomp ]
        datas = []
        for i in range(self.ncomp):
            data = np.random.randn(repeats[i]) * self.sigmas[i] + self.mus[i]
            datas.append(data)
        return np.concatenate(datas) # [ n ]
    

import numpy as np
import matplotlib.pyplot as plt

class InverseConformalPrediction:
    '''1D, non-contextual, deterministic prediction'''
    def __init__(self, mu, nres):
        self.mu = mu
        self.nres = nres

    def score(self, y):
        '''non-conformity score'''
        return np.abs(self.mu - y) # [ batch_size ]

    def get_lower_bound(self, low, high, data):
        '''
        The traditional ICP algorithm (Singh 2024)
        Args:
        - low:  lower bound of C(...)
        - high: upper bound of C(...)
        - data: [ ncal ] calibration data
        Returns:
        - coverage?
        '''
        ncal = len(data)
        scores = self.score(data) # [ ncal ]

        aa  = np.linspace(0., 1., self.nres)
        aa_ = np.clip(np.ceil((ncal+1)*aa) / ncal, a_min = None, a_max=1.)
        q   = np.quantile(scores, aa_)  # [ nres ]
        q[aa_ == 1.] = np.inf

        low_cp  = self.mu - q
        high_cp = self.mu + q

        mask = np.logical_and(low_cp > low, high_cp < high)
        try:
            index = np.where(mask == True)[0][-1]
            return aa[index]
        except:
            return 0.
        
    def get_upper_bound(self, low, high, data):
        '''
        The traditional ICP algorithm (Singh 2024)
        Args:
        - low:  lower bound of C(...)
        - high: upper bound of C(...)
        - data: [ ncal ] calibration data
        Returns:
        - coverage?
        '''
        ncal = len(data)
        scores = self.score(data) # [ ncal ]

        aa  = np.linspace(1., 0., self.nres)
        aa_ = np.clip(np.floor((ncal - 1)*aa) / ncal, a_min = 0., a_max=None)
        q   = np.quantile(scores, aa_)  # [ nres ]
        q[aa_ == 0.] = -np.inf

        low_cp  = self.mu - q
        high_cp = self.mu + q

        mask = np.logical_and(low_cp < low, high_cp > high)
        try:
            index = np.where(mask == True)[0][-1]
            return aa[index]
        except:
            return 1.

    
if __name__ == '__main__':

    # Gaussian Mixture
    import matplotlib.pyplot as plt

    kwds = {
        'mus':      np.array([1., 2.]),
        'sigmas':   np.array([0.1, 0.3])
    }

    model = GaussianMixture(**kwds)
    data = model.sample(1000)
    xx = np.linspace(0, 3, 1000)

    plt.hist(data, 100, density = True, label = 'Empriical density (histogram)')
    plt.plot(xx, model.pdf(xx), lw = 3, label = 'Density')
    plt.legend()
    plt.show()

    # Inverse conformal Prediction
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import seaborn as sns

    kwds = {
        'mu':   0,     # if mu = 10, then should get zero
        'nres': 1000
    }
    low     = -0.68
    high    = 0.68

    xx = np.linspace(low, high, 1000)
    c = norm.pdf(xx).sum() * np.diff(xx).mean()

    A, B = [], []
    for i in range(1000):
        data = np.random.randn(10) 
        model = InverseConformalPrediction(**kwds)
        a = model.get_lower_bound(low, high, data)
        b = model.get_upper_bound(low, high, data)
        A.append(a)
        B.append(b)
    sns.kdeplot(A)
    sns.kdeplot(B)
    plt.axvline(c, color = 'red', lw = 3, ls = '--')
    plt.show()