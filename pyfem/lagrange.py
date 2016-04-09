
import numpy as np

def eval_phi1d(xnodes, xref):
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

def eval_dphi1d(xnodes, xref):
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
