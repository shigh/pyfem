
import numpy as np
import scipy.sparse as sps

from .poly import gll_points
from .poly import eval_lagrange_d0, eval_lagrange_d1

class SEMhat(object):
    
    def __init__(self, order):

        n = order+1

        # GLL quadrature points
        xgll, wgll = gll_points(order+1)
        self.xgll, self.wgll = xgll, wgll
        
        # Mass matrix
        Bh = sps.dia_matrix((wgll, 0), shape=(n, n))
        self.Bh = Bh
        
        # Derivative evaluation matrix
        Dh = eval_lagrange_d1(xgll, xgll).T
        self.Dh = Dh
        
        # Diffusion operator
        # Ah = Dh'*Bh*Dh
        Ah = Dh.T.dot(Bh.dot(Dh))
        self.Ah = Ah
        
        # Convection operator
        Ch = Bh.dot(Dh)
        self.Ch = Ch

    def interp_mat(self, x):
        """Interpolation matrix from self.xgll to x

        :param x: Points to interpolate to
        :returns: Interpolation matrix
        :rtype: np.ndarray

        """

        J = eval_lagrange_d0(self.xgll, x)
        return J.T

