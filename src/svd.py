import numpy as np

def computeSVD(A, full = False):
    U, S, VT = np.linalg.svd(A, full_matrices=full)
    return U, S, VT

def dominantRatio(s):
    return (s[0]**2)/(np.sum(s**2) + 1e-12)

def cumulativeEnergy(s, threshold=1):
    s2 = s**2
    return np.sum(s2[:threshold]) / (np.sum(s2) + 1e-12)

def frobeniusErrorFromSingular(s,r):
    s2 = s**2
    if r > len(s):
        return 0.0
    return np.sqrt(np.sum(s2[r:])/np.sum(s2))

def singularSpread(s):
    norm = s/(np.sum(s) + 1e-12)
    miu = np.mean(norm)
    return np.mean((norm-miu)**2)