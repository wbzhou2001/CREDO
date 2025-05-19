import numpy as np
from shapely import Polygon, Point
from pypoman import compute_polytope_vertices
from scipy.spatial import ConvexHull
from model.sampler import GaussianMixtureSampler

class NaiveDecisionRiskAssessment:
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
        self.vertices_inverse_list  = self.get_inverse_feasible_region()
        self.model = GaussianMixtureSampler(**kwds)
        
    def get_prob(self, K, get_std = False):
        '''
        Return [ num_vertices ]
        '''
        Y_pred = self.model.sample(K)  # [ K, Y_dim ]
        # TODO: can be further paralleled
        alphas, stds = [], []
        for i in range(len(self.vertices)): # each [ Y_dim ]
            # TODO: fix this part
            mask    = Y_pred @ (self.vertices[i] - self.vertices).T >= 0 # [ K, Y_dim ] @ [ num_vertices, Y_dim ].T = [ K, num_vertices ]
            # mask    = [ 1. if Polygon(self.vertices_inverse_list[i]).contains(Point(Y_pred[k])) else 0. for k in range(K) ] # [ K ] 
            # mask    = np.array(mask)
            mask    = mask.prod(1)    # [ K ], all has to satisfy
            alpha   = mask.mean(0)    # [ 1 ]
            if get_std:
                stds.append(mask.std(0))
            alphas.append(alpha)

        if get_std:
            return alphas, stds
        else:
            return alphas
    
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
            vertices_inverse = compute_polytope_vertices(-A, b)
            vertices_inverse = np.stack(vertices_inverse)

            # correct for the ordering
            hull = ConvexHull(vertices_inverse)
            vertices_inverse = vertices_inverse[hull.vertices]

            vertices_inverse_list.append(vertices_inverse)
        return vertices_inverse_list