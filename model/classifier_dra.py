from pypoman import compute_polytope_vertices
import numpy as np
from scipy.spatial import ConvexHull
from shapely import Polygon, Point
import pandas as pd
import matplotlib.pyplot as plt 

class LinearDecisionAssessment:

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
        
        self.vertices               = self.get_feasible_region() 
        self.vertices_inverse_list  = self.get_inverse_feasible_region()
        self.model = MultiClassifier(**kwds, num_classes = len(self.vertices_inverse_list))

    def fit(self, X_tr, y_tr, fit_kwds):
        '''
        Args:
        - X_tr: [ num_tr, X_dim ]
        - y_tr: [ num_tr, num_dim ]
        '''
        y_tr_label = np.ones(y_tr.shape[0]) * 999 # 
        # convert to labels
        self.name = []
        for i in range(len(self.vertices_inverse_list)):
            v = self.vertices_inverse_list[i]
            polygon = Polygon(v)
            mask = [polygon.contains(Point(p)) for p in y_tr]
            y_tr_label[mask] = i
            self.name.append(self.vertices[i].astype(str))
        # remove outlier points
        mask = y_tr_label == 999
        y_tr_label  = y_tr_label[~mask]
        X_tr        = X_tr[~mask] 
        print(f'Removed {mask.sum()} outliers!')
        # training
        self.model.fit(X_tr, y_tr_label, **fit_kwds)

    def get_prob(self, X_te):
        '''
        Args:
        - X_te: [ num_te, X_dim ]
        '''
        y_te = self.model.get_prob(X_te)
        print(self.name)
        df = pd.DataFrame(
            index   = np.arange(len(y_te)),
            columns = np.array(self.name),
            data    = y_te 
        )
        return df
    
    def plot_feasible_region(self):
        plt.scatter(*self.vertices.T, color = 'red', label = 'Extreme points', zorder = 99) # placeholder
        patch = plt.Polygon(self.vertices, label = 'Feasible region')
        plt.gca().add_patch(patch)
        plt.title('Feasible Region')
        plt.legend()
        plt.show()

    def plot_inverse_feasible_region(self):
        for i in range(len(self.vertices_inverse_list)):
            v   = self.vertices_inverse_list[i]
            zp  = self.vertices[i] 
            plt.scatter(*v.T, color = 'none')
            patch = plt.Polygon(v, color = plt.cm.Set2(i))
            plt.gca().add_patch(patch)
            plt.text(*v.mean(0), s = zp)
        plt.title('Inverse Feasible Region')
        plt.show()
    
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
    

if __name__ == '__init__':

    A = np.array([
        [ 1,  0],   # x <= 2
        [-1,  0],   # x >= -1 → -x <= 1
        [ 0,  1],   # y <= 2
        [ 0, -1],   # y >= -1 → -y <= 1
        [ 1,  1],   # x + y <= 3
        [-1,  1],   # -x + y <= 3
    ])
    b = np.array([2, 1, 2, 1, 3, 3])

    kwds = {
        'A':    A,
        'b':    b,
        'kwds': {
            'X_dim':        1,
            'layers':       [8],
        },
        'boundary_scale':   5.
    }


    model = LinearDecisionAssessment(**kwds)


    fit_kwds = {
        'lr':   1e-0,
        'batch_size':   1000,
        'niter':    1000,
        'patience': 100
    }

    nsample = 1000

    # # random Gaussian
    # X_tr = np.random.randn(1000, 1)
    # y_tr = np.random.randn(1000, 2)

    # tilted two-mode Gaussian 
    X_tr = np.concatenate([
        np.ones(nsample//2) * 0.,
        np.ones(nsample//2) * 1.
    ], 0).reshape(-1, 1)

    y_tr = np.concatenate([
        np.random.randn(nsample//2, 2) + 1.,
        np.random.randn(nsample//2, 2) - 1.,
    ], 0)

    model.fit(X_tr, y_tr, fit_kwds)

    # get decision confidence
    X_te = np.array([0.]).reshape(1, 1)

    model.get_prob(X_te).mean(0).plot.bar(ax = plt.gca())
    plt.xticks(rotation = 30)
    plt.xlabel('Decision')
    plt.ylabel('Probability')
    plt.show()

    # model.plot_feasible_region()
    # model.plot_inverse_feasible_region()