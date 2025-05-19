import numpy as np
import torch
from torch.distributions import MultivariateNormal
from sklearn.mixture import GaussianMixture

class GaussianMixtureSampler:
    '''
    A sampler for synthetic two sample data using a series of Gaussain Mixture
    '''
    def __init__(self, mu = np.array([[-5.]]), scale = 1., weights = None, cov = None):
        '''
        Args:
        - mu:           [ ncomp, data_dim ] np, the mean of GMs
        - scale:        [ ncomp ] np, the scale of varaince of GMs 
        - noise_scale:  scaler --- optinonal, the pertubration Gaussian noise scale
        '''
        self.ncomp, self.data_dim = mu.shape
        mu     = torch.tensor(mu).float()
        if cov is None:
            cov     = torch.eye(self.data_dim).unsqueeze(-1).repeat(1, 1, self.ncomp) * torch.tensor(scale).float()   # [ data_dim, data_dim, ncomp ] th
            cov     = cov.permute(2, 0, 1) # [ ncomp, data_dim, data_dim ] th
        else:
            cov     = torch.tensor(cov).float() 
        self.dists      = [ MultivariateNormal(mu_, sigma) for mu_, sigma in zip(mu, cov) ]
        self.weights    = weights if weights is not None else np.ones(mu.shape[0]) / len(mu)

    def sample(self, nsample):
        '''
        Returns:
        - data: [ batch_size, data_dim ]
        '''
        with torch.no_grad():
            u = np.random.choice(self.ncomp, nsample, p = self.weights)
            D = []
            for i in range(len(self.dists)):
                nsample_ = (i == u).sum().item()        # scalar
                sample = self.dists[i].sample([nsample_,])  # [ nsample_, data_dim ] th
                D.append(sample.numpy())
            D = np.concatenate(D, 0)                    # [ nsample, data_dim ] np
            indices = np.random.permutation(nsample)    # [ nsample ] np 
            D = D[indices, :]                           # [ nsample, data_dim ] np
            return D                                    # [ nsample, data_dim ] np

    def pdf(self, xgrids, otype='numpy'):
        '''
        Args:
        - xgrids:   [ batch_size, data_dim ] np
        = otype:    torch or numpy --- desired return type
        Returns:
        - [ batch_size ] th or np
        '''
        if isinstance(xgrids, np.ndarray):
            xgrids = torch.tensor(xgrids).float()
        
        pdf_comps = []
        for dist in self.dists:
            pdf_comps.append(dist.log_prob(xgrids).exp()) # append [ batch_size ]

        pdf = torch.stack(pdf_comps, -1)        # [ batch_size, ncomps ] th
        pdf = (pdf * self.weights[None, :]).sum(-1)                      # [ batch_size ] th

        if otype == 'numpy':
            pdf = pdf.detach().numpy()      # [ batch_size ] np
        elif otype == 'torch':
            pass
        else:
            raise NotImplementedError('Unrecognized otype!')
        return pdf

    def fit(self, data, n_iter=100, return_kwds = True):
        '''
        Fit Gaussian mixture using sklearn and store in this sampler.
        
        Args:
        - data: [N, data_dim] np array
        - n_iter: int, number of EM iterations
        '''
        gmm = GaussianMixture(n_components=self.ncomp, 
                                max_iter=n_iter, 
                                covariance_type='full')
        gmm.fit(data)

        # Convert parameters to torch distributions
        self.weights = gmm.weights_
        mu = torch.tensor(gmm.means_).float()           # [ ncomp, data_dim ]
        cov = torch.tensor(gmm.covariances_).float()    # [ ncomp, data_dim, data_dim ]

        self.dists = [MultivariateNormal(mu[i], cov[i]) for i in range(self.ncomp)]

        if return_kwds:
            scale = np.array([ np.diag(cov[i]).mean() for i in range(len(cov))])    # [ ncomp ]
            kwds = {
                'mu':       mu.numpy(),     # [ ncomp, data_dim ] np, the mean of GMs
                'scale':    scale,  # [ ncomp ] np, the variance of GMs 
                'weights':  self.weights # [ ncomp ]
            }
            return kwds