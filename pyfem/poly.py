
import numpy as np
from scipy.special import j_roots, eval_legendre

def lagrange_list(order, roots=None):

    if roots is None:
        roots = np.linspace(-1, 1, order+1)
    assert len(roots==order+1)

    bp = []
    flags = np.ones(order+1).astype(np.bool)
    for i in range(order+1):
        flags[:] = True
        flags[i] = False
        r = roots[flags]
        c = np.prod(roots[i]-r)
        p = np.poly1d(r, True)/c
        bp.append(p)

    return bp

def legendre_list_unstable(order):
    
    k_max = order+1
    L0 = np.poly1d([1.0])
    L1 = np.poly1d([1.0, 0.0])
    x  = np.poly1d([1.0, 0.0])

    Lp = [L0, L1]
    for k in range(2, k_max+1):
        Lp.append((2.0*k-1.0)/k*x*Lp[k-1]-(k-1.0)/k*Lp[k-2])
    
    return Lp

def legendre_list(order):

    l = []
    for n in range(order+1):
        l.append(lambda x, n=n: eval_legendre(n, x))

    return l

def lobatto_list_unstable(order):
    
    k_max = order
    x  = np.poly1d([1.0, 0.0])
    l0 = (1.0-x)/2.0
    l1 = (x+1.0)/2.0

    lp = [l0, l1]
    Lp = legendre_list_unstable(k_max)
    for k in range(2, k_max+1):
        L2 = np.sqrt(2.0/(2.0*k-1.0))
        lp.append(Lp[k-1].integ()/L2)
        lp[-1] -= lp[-1](-1.0)
        
    return lp

def lobatto_list(order):
    
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
    
    x = j_roots(n-2, 1, 1)[0]
    L = eval_legendre(n-1, x)
    w1 = 2.0/(n*(n-1))
    w  = 2.0/(n*(n-1)*L*L)
    
    x = np.hstack([-1.0, x, 1.0])
    w = np.hstack([w1, w, w1])
    
    return x, w    
