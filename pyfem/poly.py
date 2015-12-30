import numpy as np

def lagrange_list(order):
    
    roots = np.linspace(-1, 1, order+1)
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

def legendre_list(order):
    
    k_max = order+1
    L0 = np.poly1d([1.0])
    L1 = np.poly1d([1.0, 0.0])
    x  = np.poly1d([1.0, 0.0])

    Lp = [L0, L1]
    for k in range(2, k_max+1):
        Lp.append((2.0*k-1.0)/k*x*Lp[k-1]-(k-1.0)/k*Lp[k-2])
    
    return Lp

def lobatto_list(order):
    
    k_max = order+1
    x  = np.poly1d([1.0, 0.0])
    l0 = (1.0-x)/2.0
    l1 = (x+1.0)/2.0

    lp = [l0, l1]
    Lp = legendre_list(k_max)
    for k in range(2, k_max+1):
        L2 = np.sqrt(2.0/(2.0*k-1.0))
        lp.append(Lp[k-1].integ()/L2)
        lp[-1] -= lp[-1](-1.0)
        
    return lp
    
