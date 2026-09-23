import numba as nb
import numpy as np

from vardyn_core.tools import csc


@nb.njit(cache=True)
def prepare_data(nobs, nwave):
    nb_by_column = np.random.randint(low=1, high=500, size=nwave)
    # Number of define data
    ng = nb_by_column.sum()
    data = np.empty(ng, dtype=np.float64)
    indices = np.empty(ng, dtype=np.uint32)
    # for each wave/column
    i = 0
    for nb in nb_by_column:
        sl = slice(i, i + nb)
        indices[sl] = np.linspace(np.random.randint(low=0, high=int(nobs / nb)), nobs - 1, nb)
        data[sl] = np.random.normal(0, 1, size=nb)
        i += nb
    return data, indices, nb_by_column


def test_transpose_matrix():
    nrow = int(1e5)
    ncol = int(nrow / 10)
    sparse_m = csc(prepare_data(nobs=nrow, nwave=ncol), (nrow, ncol))
    sparse_m_out = sparse_m.transpose().transpose()

    assert sparse_m.shape == sparse_m_out.shape
    assert (sparse_m.nb_wave_reduce == sparse_m_out.nb_wave_reduce).all()
    assert (sparse_m.indices_reduce == sparse_m_out.indices_reduce).all()
    assert (sparse_m.data == sparse_m_out.data).all()

    a = np.arange(ncol, dtype="f8")
    diff = sparse_m.dot(a) - sparse_m.transpose().tdot(a)
    assert (diff == 0).all()

    a = np.arange(nrow, dtype="f8")
    diff = sparse_m.tdot(a) - sparse_m.transpose().dot(a)
    assert (diff == 0).all()
