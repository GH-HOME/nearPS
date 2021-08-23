import numpy as np
import scipy
import scipy.sparse as sp
from scipy import sparse
from cvxopt import spmatrix

def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP


def null_dense(A, eps=1e-15):
    """
    Calculate the null space of matrix A
    eps: singular value eps, if None, use scipy nullspace
    """
    assert A is not None
    if eps is None:
        null_space = scipy.linalg.nullspace(A)
    else:
        u, s, vh = np.linalg.svd(A)
        null_space = np.compress(s <= eps, vh, axis=0).T
    return null_space

def null_dense_CVXOPT(A):
    """
    calculate the null space from sparse matrix
    """
    assert A is not None
    from cvxopt import solvers, matrix
    numx = A.shape[1]
    q = matrix(np.zeros(numx), tc='d')
    g = np.zeros(numx)
    G_cvt = matrix(np.diag(g))
    h = matrix(np.ones(numx) * 1e-6, tc='d')
    A_cvt = matrix(A)
    P = A_cvt.T * A_cvt
    solvers.options['show_progress'] = True
    sol = solvers.qp(P, q, G_cvt, h)
    null_space = np.array(sol['x']).squeeze()

    return null_space


def null_sparse_cvxopt(A, eps=1e-15):
    assert A is not None

    from cvxopt import solvers, matrix
    numx = A.shape[1]
    q = matrix(np.zeros(numx), tc='d')
    g = np.zeros(numx)
    G = sp.spdiags(g, 0, len(g), len(g))
    G_cvt = scipy_sparse_to_spmatrix(G)
    h = matrix(np.ones(numx) * 1e-6, tc='d')
    A_cvt = scipy_sparse_to_spmatrix(A)
    P = A_cvt.T * A_cvt
    solvers.options['show_progress'] = True
    sol = solvers.qp(P, q, G_cvt, h)
    null_space = np.array(sol['x']).squeeze()
    return null_space

def null_sparse(A):
    """
    calculate the null space from sparse matrix
    """
    assert A is not None
    # todo
    # add matlab version
    # add scipy version
    # add cvxopt version
    pass

def extractUpperMatrix(A, include_diag=False):
    """
    extract the upper part of the matrix, default excluding the diagonal
    :param A: N*N
    :return: vector of the upper elements
    """

    assert A.shape[0] == A.shape[1]
    size = A.shape[0]
    if include_diag:
        return A[np.triu_indices(3)]
    else:
        return A[np.triu_indices(size, k = 1)]