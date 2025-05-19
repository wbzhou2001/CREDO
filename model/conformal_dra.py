import numpy as np
from shapely import Polygon, Point
import matplotlib.pyplot as plt
from pypoman import compute_polytope_vertices
from scipy.spatial import ConvexHull
from model.sampler import GaussianMixtureSampler
import pandas as pd
from utils.setting import config

class ConformalizedDecisionRiskAssessment:
    '''
    Conformalized Decision Risk Assessment
    TODO: implement the take X as input version
    '''
    def __init__(self, A, b, kwds, boundary_scale):
        '''
        Args:
        - A:    [ num_constraints, num_dim = 2 or 3 ]
        - b:    [ num_constraints ]
        - kwds: keywords for init classifier model
        - boundary_scale:   scale of the boundary
        '''
        self.A = A
        self.b = b
        self.boundary_scale = boundary_scale
        
        self.vertices               = self.get_feasible_region()            # [ num_v, Y_dim ] np
        self.vertices_inverse_list  = self.get_inverse_feasible_region() if A.shape[1] == 2 else None
        self.model = GaussianMixtureSampler(**kwds)

    def calibrate(self, Y_cal):
        '''
        Args:
        - Y_cal:    [ num_cal, Y_dim = 2 or 3 ] np
        '''
        Y_pred = self.model.sample(Y_cal.shape[0])                  # [ num_cal, Y_dim = 2 or 3 ]
        self.E = np.linalg.norm(Y_pred - Y_cal, ord = 2, axis = 1)  # [ num_cal ]

    # def get_prob(self, K):
    #     '''
    #     '''
    #     Y_pred = self.model.sample(K)  # [ K, Y_dim ]
    #     # TODO: can be further paralleled
    #     alphas = []
    #     for i in range(len(self.vertices)): # originally [ num_v, Y_dim ]
    #         v           = self.vertices[i]                      # [ Y_dim ]
    #         other_vs    = np.delete(self.vertices, i, axis=0)   # [ num_v - 1, Y_dim ]
    #         RHS         = np.abs((v[None, :] - other_vs) @ Y_pred.T)                                # [ num_v - 1, Y_dim ] @ [ Y_dim, K ] = [ num_v - 1, K ]
    #         # RHS         = RHS / np.linalg.norm(Y_pred, ord=2, axis=1)[None, :]                    # [ num_v - 1, K ]
    #         RHS         = RHS / np.linalg.norm(v[None, :] - other_vs, ord=2, axis=1)[:, None]       # [ num_v - 1, K ]
    #         F       = (self.E[:, None] <= np.min(RHS, axis = 0)[None, :]).mean(0) # [ K ]
    #         Ncal    = len(self.E)
    #         # mask    = [ 1. if Polygon(self.vertices_inverse_list[i]).contains(Point(Y_pred[k])) else 0. for k in range(K) ] # [ K ] 
    #         # mask    = np.array(mask)
    #         mask    =  (Y_pred @ other_vs.T >= 0).prod(1) # [ K ]
    #         alpha   = ((np.floor(Ncal * F) / (Ncal + 1)) * mask).mean(0) # Directly converted to confidence
    #         alphas.append(alpha)
    #     return alphas
    
    def get_prob(self, K):
        Y_pred = self.model.sample(K)  # [ K, Y_dim ]
        # TODO: can be further paralleled
        alphas = []
        for i in range(len(self.vertices)):                     # originally [ num_v, Y_dim ]
            v           = self.vertices[i]                      # [ Y_dim ]
            other_vs    = np.delete(self.vertices, i, axis=0)   # [ num_v - 1, Y_dim ]
            norm        = (v[None, :] - other_vs) @ Y_pred.T    # [ num_v - 1, K ]

            RHS         = np.abs(norm) /    np.linalg.norm(v[None, :] - other_vs, ord=2, axis=1)[:, None] # [ num_v - 1, K ]
            F           = (self.E[:, None] <= np.min(RHS, axis = 0)[None, :]).mean(0)   # [ K ]

            mask    = (norm >= 0).prod(0) # [ K ]
            Ncal    = len(self.E)
            alpha   = ((np.floor(Ncal * F) / (Ncal + 1)) * mask).mean(0) # Directly converted to confidence
            alphas.append(alpha)
        return alphas
    
    def plot_feasible_region(self, ax, kwds = None, vertice = False):
        # ax.scatter(*self.vertices.T, color = 'red', label = 'Vertices', zorder = 99, s = 80) # placeholder
        ax.scatter(*self.vertices.T, color = 'none' if not vertice else 'red', label = 'Vertices', zorder = 99, s = 80) # placeholder
        if kwds is None:
            patch = plt.Polygon(self.vertices, label = 'Feasible region', facecolor = 'lightgray', edgecolor = 'none', lw = 2)
        else:
            patch = plt.Polygon(self.vertices, **kwds)
        ax.add_patch(patch)

        # names = 'ABCDEFG'
        # for i in range(len(self.vertices)):
        #     v = self.vertices[i]
        #     ax.text(v[0], v[1] + 0.2, s = 'Decision ' + names[i], va = 'bottom', ha = 'center', color= 'red')
        return ax

    def plot_inverse_feasible_region(self, ax, kwds = None):
        for i in range(len(self.vertices_inverse_list)):
            v   = self.vertices_inverse_list[i]
            ax.scatter(*v.T, color = 'none')
            if kwds is None:
                patch = plt.Polygon(v, facecolor = 'none', lw = 2, ls = '-', edgecolor = 'red', zorder = 99)
            else:
                patch = plt.Polygon(v, **kwds)
            ax.add_patch(patch)
        return ax
    
    def get_feasible_region(self):
        '''
        return the vertices of the (single) polygon
        '''
        vertices = compute_polytope_vertices(self.A, self.b)
        vertices = np.stack(vertices)

        # correct for the ordering
        hull = ConvexHull(vertices)
        vertices = vertices[hull.vertices]
        return vertices # [ nv, num_dim ] np
    
    def get_inverse_feasible_region(self):
        '''
        Args:
        - [ nv, num_dim ] np
        Returns a list of all vertices of polygons
        '''
        vertices = self.get_feasible_region()
        vertices_inverse_list = []
        for zp in vertices:
            A = vertices - zp
            b = np.zeros(A.shape[0])

            # correction for boundary 
            A_ = np.array([[-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
            b_ = np.array([1., 1., 1., 1.]) * self.boundary_scale
            A = np.concatenate([A, A_], 0)
            b = np.concatenate([b, b_], 0)
            vertices_inverse = compute_polytope_vertices(A, b)
            vertices_inverse = np.stack(vertices_inverse)

            # correct for the ordering
            hull = ConvexHull(vertices_inverse)
            vertices_inverse = vertices_inverse[hull.vertices]

            vertices_inverse_list.append(vertices_inverse)
        return vertices_inverse_list