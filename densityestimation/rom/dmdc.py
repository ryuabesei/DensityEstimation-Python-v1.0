import numpy as np
from scipy.linalg import expm


def discretize_lin(Ac, Bc, dt):
    """
    連続時間系:
      dot{z} = Ac z + Bc u
    をゼロ次ホールドで離散化:
      z_{k+1} = Ad z_k + Bd u_k
    """
    n = Ac.shape[0]
    M = np.zeros((n+n, n+n))
    M[:n,:n] = Ac
    M[:n,n:] = Bc
    Md = expm(M * dt)
    Ad = Md[:n,:n]
    Bd = Md[:n,n:]
    return Ad, Bd
