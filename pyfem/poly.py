
import numpy as np
from scipy.special import j_roots, eval_legendre

def legendre_list(order):

    l = []
    for n in range(order+1):
        l.append(lambda x, n=n: eval_legendre(n, x))

    return l

def lobatto_list(order):
    """List of lobatto polynomials

    :param order: Highest polynomial order
    :returns: List of polynomial functions
    :rtype:

    """
    
    x  = np.poly1d([1.0, 0.0])
    l0 = (1.0-x)/2.0
    l1 = (x+1.0)/2.0

    lp = [l0, l1]
    Lp = legendre_list(order)
    el = eval_legendre
    for k in range(2, order+1):
        # See the wiki article on Legendre polys and the high order
        # finite elements book for details on the constants
        a = np.sqrt((2.*k-1.)/2.)/(2.*(k-1.)+1.)
        lp.append(lambda x,a=a,k=k: (el((k-1)+1,x)-el((k-1)-1, x))*a)

    return lp

def lobatto_list_d1(order):
    """List of first derivative of lobatto polynomials

    :param order: Highest polynomial order
    :returns: List of polynomial functions
    :rtype:

    """
    
    lp = [lambda x:-.5+x*0,
          lambda x:+.5+x*0]
    el = eval_legendre
    for k in range(2, order+1):
        a = np.sqrt((2.*k-1.)/2.)
        lp.append(lambda x,a=a,k=k: el(k-1,x)*a)

    return lp

def gll_points(n):
    """GLL points and weights

    :param n: Number of points
    :returns: (x, w)
    :rtype: 

    """
    
    assert n>=2

    if n==2:
        x = np.array([-1.0, 1.0])
        w = np.array([ 1.0, 1.0])
        return x, w

    # See Nodal Discontinuous Galerkin Methods Appendix A for x and
    # the Mathworld page on Lobatto Quadrature for w
    x = j_roots(n-2, 1, 1)[0]
    L = eval_legendre(n-1, x)
    w1 = 2.0/(n*(n-1))
    w  = 2.0/(n*(n-1)*L*L)
    
    x = np.hstack([-1.0, x, 1.0])
    w = np.hstack([w1, w, w1])
    
    return x, w    

def eval_lagrange_d0(xnodes, xref):
    """Stable lagrange eval on interval ref

    :param xnodes: Lagrange nodes
    :param xref: Ref locations to eval
    :returns: phi at ref points for each basis function
    :rtype: np.ndarray: len(nxnodes) x len(xref)

    """

    nxnodes = len(xnodes)
    nx      = len(xref)

    phi1d = np.ones((nxnodes,nx),dtype=float)
    coeff = np.zeros((nxnodes,))
    for i in range(0,nxnodes):

        # compute (xi-x1)(xi-x2)...(xi-xn) for each x
        coeff[i]=1.0
        for j in range(0,nxnodes):
            if i!=j:
                coeff[i] *= xnodes[i] - xnodes[j]
                
        for k in range(0,nx):
            for j in range(0,nxnodes):
                if i!=j:
                    phi1d[i,k] *= xref[k] - xnodes[j]
            phi1d[i,k] *= 1.0/coeff[i]
            
    return phi1d

def eval_lagrange_d1(xnodes, xref):
    """Stable lagrange derivative eval on interval ref

    :param xnodes: Lagrange nodes
    :param xref: Ref locations to eval
    :returns: phi at ref points for each basis function
    :rtype: np.ndarray: len(nxnodes) x len(xref)

    """

    nxnodes = len(xnodes)
    nx      = len(xref)

    dphi1d = np.zeros((nxnodes,nx),dtype=float)
    coeff  = np.zeros((nxnodes,))
    for i in range(0,nxnodes):

        # compute (xi-x1)(xi-x2)...(xi-xn) for each x
        coeff[i]=1.0
        for j in range(0,nxnodes):
            if i!=j:
                coeff[i] *= xnodes[i] - xnodes[j]
                
        # comput d l_i at xg
        for k in range(0,nx):
            for j in range(0,nxnodes):
                if i!=j:
                    addon = 1.0
                    for l in range(0,nxnodes):
                        if (l!=j) and (l!=i):
                            addon *= xref[k]-xnodes[l]
                    dphi1d[i,k] += addon
            dphi1d[i,k] *= 1.0/coeff[i]

    return dphi1d
    
