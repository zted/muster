"""
This file contains mathematical functions useful for image representation data
"""

from math import log

import numpy as np
import scipy.spatial as ss
from scipy.special import digamma


def entropy(x, k=3, base=2):
    """
        Adapted from Greg Ver Steeg's NPEET toolkit - more info http://www.isi.edu/~gregv/npeet.html
        The classic K-L k-nearest neighbor continuous entropy estimator
        :param x: a list of numbers, e.g. x = [1.3,3.7,5.1,2.4]
        :param k: lower bound on how many elements must be in x
        :param base: base to work in
        :return:
        """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = [[elem] for elem in x]
    d = len(x[0])
    N = len(x)
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * np.random.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]
    const = digamma(N) - digamma(k) + d * log(2)
    return (const + d * np.mean(map(log, nn))) / log(base)


def computationsPerDimension(vecs):
    dimensions = len(vecs[0])
    mean = np.array([0.0] * dimensions)
    std = mean.copy()
    ent = mean.copy()
    mp = mean.copy()
    # ^mp stands for maxpool, we take the maximums of each vector
    for i in range(dimensions):
        allNums = np.array([j[i] for j in vecs])
        mean[i] = allNums.mean()
        std[i] = allNums.std()
        ent[i] = entropy(allNums)
        mp[i] = allNums.max()
    return mean, std, ent, mp


def calculateDispersion(vecs):
    """
        :param vecs: list of vector representations of a concept word
        :return: the dispersion of said concept word. a scalar
        """
    numVecs = len(vecs)
    accum = 0.0
    for i in range(numVecs - 1):
        for j in range(i + 1, numVecs):
            vi = vecs[i];
            vj = vecs[j]
            dp = np.dot(vi, vj)
            denom = np.linalg.norm(vi) * np.linalg.norm(vj)
            accum += (1 - dp / denom)
    dispersion = accum / (2.0 * numVecs * (numVecs - 1))
    return dispersion


def meanEntropy(vec):
    """
        :param vecs: a single vector of numbers
        :return: a scalar, which is the entropy of the vector
        """
    n = np.linalg.norm(vec)
    accum = 0
    for v in vec:
        p_norm = v / n
        if p_norm == 0:
            result = 0
        else:
            result = p_norm * np.log2(p_norm)
        accum += result
    return -accum
