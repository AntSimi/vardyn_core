"""
Provides shared numerical, interpolation, and filtering utilities.
"""

import logging

import numpy as np
import scipy
from scipy import interpolate, spatial
import scipy.linalg as spl
import xarray as xr

logger = logging.getLogger("vardyn")


class GridInterpolator:
    def __init__(self, da, **kwargs):
        self.func = interpolate.RegularGridInterpolator([da[i].values for i in da.dims], da.values, **kwargs)
        self.dims = da.dims

    def __call__(self, **kwargs):
        coords = np.meshgrid(*[kwargs[i] for i in self.dims])
        return self.func(np.column_stack([v.ravel() for v in coords]))


def gaspari_cohn(r, c=1):
    """
    NAME
        bfn_gaspari_cohn

    DESCRIPTION
        Gaspari-Cohn function. Inspired from E.Cosmes.

        Args:
            r : array of value whose the Gaspari-Cohn function will be applied
            c : Distance above which the return values are zeros


        Returns:  smoothed values

    """
    if type(r) in [float, int, np.float32, np.float64]:
        ra = np.array([r])
    else:
        ra = r
    if c <= 0:
        return np.zeros_like(ra)
    else:
        ra = 2 * np.abs(ra) / c
        gp = np.zeros_like(ra)
        i = np.where(ra <= 1.0)[0]
        gp[i] = -0.25 * ra[i] ** 5 + 0.5 * ra[i] ** 4 + 0.625 * ra[i] ** 3 - 5.0 / 3.0 * ra[i] ** 2 + 1.0
        i = np.where((ra > 1.0) * (ra <= 2.0))[0]
        gp[i] = (
            1.0 / 12.0 * ra[i] ** 5 - 0.5 * ra[i] ** 4 + 0.625 * ra[i] ** 3 + 5.0 / 3.0 * ra[i] ** 2 - 5.0 * ra[i] + 4.0 - 2.0 / 3.0 / ra[i]
        )
        if type(r) is float:
            gp = gp[0]
    return gp


def lonlat2dxdy(lon, lat):
    dlon = np.gradient(lon)
    dlat = np.gradient(lat)
    dx = np.sqrt((dlon[1] * 111000 * np.cos(np.deg2rad(lat))) ** 2 + (dlat[1] * 111000) ** 2)
    dy = np.sqrt((dlon[0] * 111000 * np.cos(np.deg2rad(lat))) ** 2 + (dlat[0] * 111000) ** 2)
    dx[0, :] = dx[1, :]
    dx[-1, :] = dx[-2, :]
    dx[:, 0] = dx[:, 1]
    dx[:, -1] = dx[:, -2]
    dy[0, :] = dy[1, :]
    dy[-1, :] = dy[-2, :]
    dy[:, 0] = dy[:, 1]
    dy[:, -1] = dy[:, -2]

    return dx, dy


def dxdy2xy(dx, dy, x0=0, y0=0):
    ny, nx = dx.shape
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            X[i, j] = x0 + np.sum(dx[i, :j])
            Y[i, j] = y0 + np.sum(dy[:i, j])
    return X, Y


def geo2cart(coords):
    """
    NAME
        geo2cart

    DESCRIPTION
        Transform coordinates from geodetic to cartesian

        Args:
            coords : a set of lan/lon coordinates (e.g. a tuple or
             an array of tuples)


        Returns: a set of cartesian coordinates (x,y,z)

    """

    A = 6378.137
    E2 = 6.69437999014e-3

    coords = np.asarray(coords).astype(float)

    if coords.ndim == 1:
        coords = np.array([coords])

    lat_rad = np.radians(coords[:, 1])
    lon_rad = np.radians(coords[:, 0])

    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)

    return np.column_stack((x, y, z))


def interp2d(ds, name_vars, lon_out, lat_out):
    logger.warning("Must be replace by RegularGridInterpolator")

    ds = ds.squeeze()

    ds = ds.assign_coords({name_vars["lon"]: (ds[name_vars["lon"]]), name_vars["lat"]: ds[name_vars["lat"]]})

    if ds[name_vars["var"]].shape[0] != ds[name_vars["lat"]].shape[0]:
        ds[name_vars["var"]] = ds[name_vars["var"]].transpose()

    if len(ds[name_vars["lon"]].shape) == 2:
        dlon = (ds[name_vars["lon"]][:, 1:].values - ds[name_vars["lon"]][:, :-1].values).max()
        dlat = (ds[name_vars["lat"]][1:, :].values - ds[name_vars["lat"]][:-1, :].values).max()

        ds = ds.where(
            (ds[name_vars["lon"]] <= lon_out.max() + dlon)
            & (ds[name_vars["lon"]] >= lon_out.min() - dlon)
            & (ds[name_vars["lat"]] <= lat_out.max() + dlat)
            & (ds[name_vars["lat"]] >= lat_out.min() - dlat),
            drop=True,
        )

        lon_sel = ds[name_vars["lon"]].values
        lat_sel = ds[name_vars["lat"]].values

    else:
        dlon = (ds[name_vars["lon"]][1:].values - ds[name_vars["lon"]][:-1].values).max()
        dlat = (ds[name_vars["lat"]][1:].values - ds[name_vars["lat"]][:-1].values).max()

        ds = ds.where(
            (ds[name_vars["lon"]] <= lon_out.max() + dlon)
            & (ds[name_vars["lon"]] >= lon_out.min() - dlon)
            & (ds[name_vars["lat"]] <= lat_out.max() + dlat)
            & (ds[name_vars["lat"]] >= lat_out.min() - dlat),
            drop=True,
        )

        lon_sel, lat_sel = np.meshgrid(ds[name_vars["lon"]].values, ds[name_vars["lat"]].values)

    var_sel = ds[name_vars["var"]].values

    var_out = interpolate.griddata((lon_sel.ravel(), lat_sel.ravel()), var_sel.ravel(), (lon_out.ravel(), lat_out.ravel())).reshape(
        (lat_out.shape)
    )

    return var_out


def compute_weight_map(lon2d, lat2d, mask, dist_scale, bc=True, slope=10):

    coords = np.column_stack((lon2d.ravel(), lat2d.ravel()))
    coords_cart = geo2cart(coords)
    ground_pixel_tree = spatial.cKDTree(coords_cart)
    n_probe = min(100, len(coords_cart))
    dd, _ = ground_pixel_tree.query(coords_cart[:n_probe], k=2)
    dist_threshold = dd[:, 1].min()
    mask = mask.copy()
    if bc:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

    lon_bc = lon2d[mask]
    lat_bc = lat2d[mask]
    coords_bc = np.column_stack((lon_bc, lat_bc))
    bc_tree = spatial.cKDTree(geo2cart(coords_bc))

    dist_to_bc, _ = bc_tree.query(coords_cart, k=1)
    dist_to_bc = np.maximum(dist_to_bc - 0.5 * dist_threshold, 0.0)

    x = np.clip(dist_to_bc / dist_scale, 0.0, 1.0)
    bc_weight = 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    bc_weight = bc_weight.reshape(lon2d.shape)

    return bc_weight


def compute_sponge_components(lon2d, lat2d, mask, dist_scale):
    """Compute separate sponge weight maps for coast, N/S edges, and W/E edges."""

    w_coast = compute_weight_map(lon2d, lat2d, mask, dist_scale, bc=False)

    mask_NS = np.zeros_like(mask)
    mask_NS[0, :] = True
    mask_NS[-1, :] = True
    w_NS = compute_weight_map(lon2d, lat2d, mask_NS, dist_scale, bc=False)

    mask_WE = np.zeros_like(mask)
    mask_WE[:, 0] = True
    mask_WE[:, -1] = True
    w_WE = compute_weight_map(lon2d, lat2d, mask_WE, dist_scale, bc=False)

    return w_coast, w_NS, w_WE


def smooth_weight_map(lon2d, lat2d, mask, dist_scale):
    """Isotropic sponge weight with smooth corner blending."""

    w_coast, w_NS, w_WE = compute_sponge_components(lon2d, lat2d, mask, dist_scale)
    return 1.0 - (1.0 - w_coast) * (1.0 - w_NS) * (1.0 - w_WE)


def detrendn(da, axes=None):
    """
    Detrend by subtracting out the least-square plane or least-square cubic fit
    depending on the number of axis.
    Parameters
    ----------
    da : `dask.array`
        The data to be detrended
    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """

    if axes is None:
        axes = range(len(da.shape))

    N = [da.shape[n] for n in axes]
    M = []
    for n in range(da.ndim):
        if n not in axes:
            M.append(da.shape[n])

    if len(N) == 2:
        G = np.ones((N[0] * N[1], 3))
        for i in range(N[0]):
            G[N[1] * i : N[1] * i + N[1], 1] = i + 1
            G[N[1] * i : N[1] * i + N[1], 2] = np.arange(1, N[1] + 1)
        if isinstance(da, xr.DataArray):
            d_obs = np.reshape(da.copy().values, (N[0] * N[1], 1))
        else:
            d_obs = np.reshape(da.copy(), (N[0] * N[1], 1))
    elif len(N) == 3:
        if isinstance(da, xr.DataArray):
            if da.ndim > 3:
                raise NotImplementedError(
                    "Cubic detrend is not implemented "
                    "for 4-dimensional `xarray.DataArray`."
                    " We suggest converting it to "
                    "`dask.array`."
                )
            else:
                d_obs = np.reshape(da.copy().values, (N[0] * N[1] * N[2], 1))
        else:
            d_obs = np.reshape(da.copy(), (N[0] * N[1] * N[2], 1))

        G = np.ones((N[0] * N[1] * N[2], 4))
        G[:, 3] = np.tile(np.arange(1, N[2] + 1), N[0] * N[1])
        ys = np.zeros(N[1] * N[2])
        for i in range(N[1]):
            ys[N[2] * i : N[2] * i + N[2]] = i + 1
        G[:, 2] = np.tile(ys, N[0])
        for i in range(N[0]):
            G[len(ys) * i : len(ys) * i + len(ys), 1] = i + 1
    else:
        raise NotImplementedError("Detrending over more than 4 axes is " "not implemented.")

    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    lin_trend = np.reshape(d_est, da.shape)

    return da - lin_trend


def read_auxdata(file_aux, name_var, lon_unit):

    # Read database
    ds = xr.open_dataset(file_aux, decode_times=False)
    if np.sign(ds[name_var["lon"]].data.min()) == -1 and lon_unit == "0_360":
        ds = ds.assign_coords({name_var["lon"]: ((name_var["lon"], ds[name_var["lon"]].data % 360))})
    elif (np.sign(ds[name_var["lon"]].data.min()) == 1 or ds[name_var["lon"]].data.max() > 180) and lon_unit == "-180_180":
        ds = ds.assign_coords({name_var["lon"]: ((name_var["lon"], (ds[name_var["lon"]].data + 180) % 360 - 180))})
    ds = ds.sortby(ds[name_var["lon"]])

    lon = ds[name_var["lon"]].values
    lat = ds[name_var["lat"]].values
    for k, v in name_var.items():
        if k in ("lon", "lat"):
            continue
        data = ds[v].values.squeeze()

    if len(np.shape(data)) == 3:
        # FIXME on ne tien pas compte de la variabilité saisoniere du bruit
        data = np.mean(data, 0)

    if data.shape[1] == lon.size:
        data = data.transpose()

    if len(lon.shape) == 1:
        finterp = scipy.interpolate.RegularGridInterpolator((lon, lat), data, bounds_error=False, fill_value=None)
    else:
        finterp = scipy.interpolate.LinearNDInterpolator(list(zip(lon.ravel(), lat.ravel())), data.ravel())
    return finterp


def ssh2uv(ssh, State=None, lon=None, lat=None, xac=None, g=9.81):

    if lon is not None and lat is not None:
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        f = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
        dx, dy = lonlat2dxdy(lon, lat)
    else:
        f = State.f
        dx = State.DX
        dy = State.DY

    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        _dx = dx
        _dy = dy
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f = f[:, :, np.newaxis]
        _dx = dx[:, :, np.newaxis]
        _dy = dy[:, :, np.newaxis]

    u = np.zeros_like(ssh) * np.nan
    v = np.zeros_like(ssh) * np.nan
    u[1:-1, 1:] = -g / f[1:-1, 1:] * (ssh[2:, :-1] + ssh[2:, 1:] - ssh[:-2, 1:] - ssh[:-2, :-1]) / (4 * _dy[1:-1, 1:])
    v[1:, 1:-1] = +g / f[1:, 1:-1] * (ssh[1:, 2:] + ssh[:-1, 2:] - ssh[:-1, :-2] - ssh[1:, :-2]) / (4 * _dx[1:, 1:-1])

    if xac is not None:
        u = _masked_edge(u, xac)
        v = _masked_edge(v, xac)

    if ssh_shapelen == 3:
        u = np.moveaxis(u, -1, 0)
        v = np.moveaxis(v, -1, 0)

    return u, v


def ssh2rv(ssh, State=None, lon=None, lat=None, xac=None, g=9.81, norm=False):

    if lon is not None and lat is not None:
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        f = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
        dx, dy = lonlat2dxdy(lon, lat)
    else:
        f = State.f
        dx = State.DX
        dy = State.DY

    ssh_shapelen = len(ssh.shape)
    if ssh_shapelen == 2:
        _dx = dx[1:-1, 1:-1]
        _dy = dy[1:-1, 1:-1]
    elif ssh_shapelen == 3:
        ssh = np.moveaxis(ssh, 0, -1)
        f = f[:, :, np.newaxis]
        _dx = dx[1:-1, 1:-1, np.newaxis]
        _dy = dy[1:-1, 1:-1, np.newaxis]

    rv = np.zeros_like(ssh) * np.nan
    rv[1:-1, 1:-1] = (
        g
        / f[1:-1, 1:-1]
        * (
            (ssh[2:, 1:-1] + ssh[:-2, 1:-1] - 2 * ssh[1:-1, 1:-1]) / _dy**2
            + (ssh[1:-1, 2:] + ssh[1:-1, :-2] - 2 * ssh[1:-1, 1:-1]) / _dx**2
        )
    )
    if norm:
        rv /= f

    if xac is not None:
        rv = _masked_edge(rv, xac)

    if ssh_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv


def uv2rv(UV, State=None, lon=None, lat=None, xac=None, norm=False):
    try:
        dx, dy = lonlat2dxdy(lon, lat)
    except Exception:
        dx = State.DX
        dy = State.DY

    u = UV[0]
    v = UV[1]

    uv_shapelen = len(u.shape)
    if uv_shapelen == 2:
        _dx = dx
        _dy = dy
    elif uv_shapelen == 3:
        u = np.moveaxis(u, 0, -1)
        v = np.moveaxis(v, 0, -1)
        _dx = dx[:, :, np.newaxis]
        _dy = dy[:, :, np.newaxis]

    mask = np.isnan(u)
    rv = np.gradient(v, axis=1) / _dx - np.gradient(u, axis=0) / _dy

    if norm:
        f = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)
        rv /= f

    if xac is not None:
        rv = _masked_edge(rv, xac)

    rv[mask] = np.nan

    if uv_shapelen == 3:
        rv = np.moveaxis(rv, -1, 0)

    return rv


def _masked_edge(var, xac):
    """Mask the edges of the swath gap."""

    if np.any(xac > 0):
        ind_gap = xac == np.nanmin(xac[xac > 0])
        if ind_gap.size == var.size:
            if ind_gap.shape != var.shape:
                ind_gap = ind_gap.transpose()
            var[ind_gap] = np.nan
        elif ind_gap.size == var.shape[1]:
            var[:, ind_gap] = np.nan
    if np.any(xac < 0):
        ind_gap = xac == np.nanmax(xac[xac < 0])
        if ind_gap.size == var.size:
            if ind_gap.shape != var.shape:
                ind_gap = ind_gap.transpose()
            var[ind_gap] = np.nan
        elif ind_gap.size == var.shape[1]:
            var[:, ind_gap] = np.nan

    return var


import numba


def reduce_indices(nb_wave, indices, n_wave):
    indices_ = np.empty(indices.size // n_wave, dtype=indices.dtype)
    i = 0
    for n in nb_wave:
        if n == 0:
            continue
        indices_[i : i + n] = indices[i * n_wave : i * n_wave + n]
        i += n
    return indices_


@numba.njit(cache=True)
def dot_wave_reduce_add(data, indices_reduce, nb_wave_reduce, wave_repetition, v, out):
    # v est de la taille du nombre de colonne
    # la sortie est de la taille du nombre de colonne
    M = nb_wave_reduce.size
    k = 0
    i_data = 0
    for i in range(M):
        nb_wave_ = nb_wave_reduce[i]
        i0, i1 = k, k + nb_wave_
        k += nb_wave_
        if nb_wave_ == 0:
            continue
        indices_reduce_ = indices_reduce[i0:i1]
        out_ = np.zeros(nb_wave_, dtype=out.dtype)
        for l in range(wave_repetition):
            v_ = v[i * wave_repetition + l]
            if v_ != 0:
                data_ = data[i_data : i_data + nb_wave_]
                for j in range(nb_wave_):
                    out_[j] += data_[j] * v_
            i_data += nb_wave_
        for j in range(nb_wave_):
            out[indices_reduce_[j]] += out_[j]


@numba.njit(cache=True)
def tdot_wave_reduce_add(data, indices_reduce, nb_wave_reduce, wave_repetition, v, out):
    """Transpose and dot in same operation."""
    # v est de la taille du nombre de ligne
    i0 = 0
    i_data = 0
    # iteration sur les colonnes
    for i in range(nb_wave_reduce.size):
        nb_wave_ = nb_wave_reduce[i]
        if nb_wave_:
            # Iteration sur N colonne contenant les meme indices
            indices_reduce_ = indices_reduce[i0 : i0 + nb_wave_]
            i_start = i * wave_repetition
            for l in range(i_start, i_start + wave_repetition):
                out_ = 0
                data_ = data[i_data : i_data + nb_wave_]
                for j in range(nb_wave_):
                    out_ += v[indices_reduce_[j]] * data_[j]
                out[l] += out_
                i_data += nb_wave_
            i0 += nb_wave_


@numba.njit(cache=True)
def build_indices(ir, wr, wave_repetition, indices):
    i, j = 0, 0
    for n in wr:
        if n == 0:
            continue
        for _ in range(wave_repetition):
            indices[j : j + n] = ir[i : i + n]
            j += n
        i += n
    return indices


class csc:
    __slots__ = ("data", "indices_reduce", "nb_wave_reduce", "shape", "dtype", "wave_repetition")

    def __repr__(self):
        return f"Sparse matrix {self.shape[0]:_} x {self.shape[1]:_}  with {self.nnz:_} nnz loaded {self.nnz/(self.shape[0] * self.shape[1]) * 100:.2f}%"

    def __sizeof__(self):
        """Retrun only size of matrix without littles attributes."""
        return self.data_size + self.index_size

    @staticmethod
    def reduce_indices(nb_wave, indices, n_wave):
        if nb_wave is None:
            return None, None
        nb_wave = nb_wave[::n_wave].copy()
        indices_ = reduce_indices(nb_wave, indices, n_wave)
        return nb_wave, indices_

    def __init__(self, datas, shape, dtype="f4", wave_repetition=1):
        self.wave_repetition = wave_repetition
        self.data, indices, nb_wave = datas
        self.nb_wave_reduce, indices_reduce = self.reduce_indices(nb_wave, indices, self.wave_repetition)
        self.indices_reduce = indices_reduce.astype(np.min_scalar_type(indices_reduce.max()))
        self.shape = shape
        self.dtype = dtype
        if self.data is not None:
            if not np.issubdtype(self.data.dtype, dtype):
                self.data = self.data.astype(dtype)

    @property
    def nb_wave(self):
        return self.nb_wave_reduce[:].repeat(self.wave_repetition)

    @property
    def indices(self):
        indices_ = np.empty(self.data.size, dtype=self.indices_reduce.dtype)
        build_indices(self.indices_reduce[:], self.nb_wave_reduce[:], self.wave_repetition, indices_)
        return indices_

    @property
    def nnz(self):
        if self.data is None:
            return 0
        return self.data.size

    def tdot_add(self, v, out):
        """Transpose and dot in same operation."""
        tdot_wave_reduce_add(self.data[:], self.indices_reduce[:], self.nb_wave_reduce[:], self.wave_repetition, v, out)

    def tdot(self, v):
        """Transpose and dot in same operation."""
        out = np.zeros(self.shape[1], dtype=v.dtype)
        self.tdot_add(v, out)
        return out

    def dot_add(self, v, out):
        dot_wave_reduce_add(self.data[:], self.indices_reduce[:], self.nb_wave_reduce[:], self.wave_repetition, v, out)

    def dot(self, v):
        out = np.zeros(self.shape[0], dtype=v.dtype)
        self.dot_add(v, out)
        return out

    def transpose(self):
        return self.__class__(
            transpose_sparse_matrix(self.data, self.indices, self.nb_wave), (self.shape[1], self.shape[0]), dtype=self.dtype
        )


@numba.njit(cache=True)
def transpose_sparse_matrix(data, indices, nb_by_column):
    nb = data.shape[0]
    new_data = np.empty(nb, dtype=data.dtype)
    new_indices = np.empty(nb, dtype=nb_by_column.dtype)

    i_column = np.empty(nb_by_column.size + 1, dtype="u4")
    i_column[0] = 0
    i_column[1:] = nb_by_column.cumsum()
    nb_by_row = np.bincount(indices)
    i_c, i_c_stop = 0, i_column[1]
    row_cur = nb_by_row.cumsum() - nb_by_row
    for i in range(nb):
        i_row = indices[i]
        new_i = row_cur[i_row]
        new_data[new_i] = data[i]
        while i >= i_c_stop:
            i_c += 1
            i_c_stop = i_column[i_c + 1]
        new_indices[new_i] = i_c
        row_cur[i_row] += 1
    return new_data, new_indices, nb_by_row
