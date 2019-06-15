import numpy as np


def pca(x, y, num_components=0):
    # [n, d] = x.shape
    y = np.asarray(y)
    [n, d] = x.shape, y.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = x.mean(axis=0)
    x = x - mu
    if n > d:
        c = np.dot(x.T, x)
        [eigenvalues, eigenvectors] = np.linalg.eigh(c)
    else:
        c = np.dot(x, x.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(c)
        eigenvectors = np.dot(x.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np. linalg.svd (x.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # select only num_components
    eigenvalues = eigenvalues[0: num_components[0]].copy()  # num_components[0 or 1]
    eigenvectors = eigenvectors[:, 0: num_components[0]].copy()  # num_components[0 or 1]
    return [eigenvalues, eigenvectors, mu]


def project(w, x, mu=None):
    if mu is None:
        return np.dot(x, w)
    return np.dot(x - mu, w)


def reconstruct(w, y, mu=None):
    if mu is None:
        return np.dot(y, w.T)
    return np.dot(y, w .T) + mu



