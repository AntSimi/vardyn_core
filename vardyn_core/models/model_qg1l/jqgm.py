"""
Implements the 1.5-layer quasi-geostrophic model and variants.
"""

from functools import partial

import jax
from jax import jit, jvp, vjp
from jax.lax import scan
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.sparse.linalg import cg as jcg
import numpy as np

from ...config import USE_FLOAT64

jax.config.update("jax_enable_x64", USE_FLOAT64)


def scan_python(f, init, xs=None, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)


def dynamic_slice_tile(array, y_start, x_start, tile_height, tile_width):
    """
    Dynamically slice a tile from the array.

    Args:
        array (jax.numpy.ndarray): Input array to slice from.
        y_start (int): Starting y-index.
        x_start (int): Starting x-index.
        tile_height (int): Height of the tile.
        tile_width (int): Width of the tile.

    Returns:
        jax.numpy.ndarray: The sliced tile.
    """
    return jax.lax.dynamic_slice(array, (y_start, x_start), (tile_height, tile_width))


def dstI1D(x, norm="ortho"):
    """1D type-I discrete sine transform."""
    nx, ny = x.shape
    x_ = jnp.zeros((nx, ny + 2))
    x_ = x_.at[:, 1:-1].set(x)
    res = jnp.fft.irfft(-1j * x_, axis=-1, norm=norm)
    return res[:, 1 : ny + 1]


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).T, norm=norm).T


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    nx, ny = x.shape
    x_ = jnp.zeros((nx + 2, ny + 2), dtype=jnp.complex64)
    x_ = x_.at[1:-1, 1:-1].set(-1j * x)
    res = jnp.fft.irfft(x_, axis=-1, norm=norm)
    # CHECK suspicious res[: nx + 2] => but result is okay
    res = jnp.fft.irfft(-1j * res[: nx + 2, 1 : ny + 1], axis=0, norm=norm)
    return res[1 : nx + 1]


@jit
def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    if USE_FLOAT64:
        _f = f.astype("float64")
    else:
        _f = f.astype("float32")
    return dstI2D(dstI2D(_f) / operator_dst)


@jit
def inverse_elliptic_dst_tgl(dh0, h0):
    _, dh1 = jvp(inverse_elliptic_dst, (h0,), (dh0,))

    return dh1


@jit
def inverse_elliptic_dst_adj(adh0, h0):
    _, adf = vjp(inverse_elliptic_dst, h0)

    return adf(adh0)[0]


def cg(q_flat, h2pv_operator, tol=1e-5, maxiter=1000):
    return jcg(h2pv_operator, q_flat, tol=tol, maxiter=maxiter)


def diffusion(v, dx, dy):
    k = jnp.array(
        [
            [0, 1 / dy**2, 0],
            [1 / dx**2, -2 / dy**2 - 2 / dx**2, 1 / dx**2],
            [0, 1 / dy**2, 0],
        ]
    )
    return Qgm.conv(v[1:-1, 1:-1], k)
    # hk = jnp.array([[1,-2,1]])/ dx**2
    # vk = jnp.array([[1],[-2],[1]])/ dy**2
    # return Qgm.hconv(v[:,1:-1], hk) + Qgm.vconv(v[1:-1], vk)


class Qgm:
    NAME = "MOD_QG1L"

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(
        self,
        dx=None,
        dy=None,
        dt=None,
        SSH=None,
        c=None,
        Kdiffus=None,
        upwind=3,
        g=9.81,
        f=1e-4,
        time_scheme="Euler",
        compile=True,
        Wbc=None,
        Kdiffus_trac=None,
        bc_trac="OBC",
        mdt=None,
        ageo_velocities=False,
        advect_pv=True,
        sponge_coef=0.0,
        bathymetry_PV_term=None,
        formulation="ssh",
        **kwargs,
    ):

        if USE_FLOAT64:
            self.dtype = "float64"
        else:
            self.dtype = "float32"

        # Formulation: 'ssh' (work in SSH space) or 'sf' (work in streamfunction space)
        self.formulation = formulation

        # Grid shape
        (
            ny,
            nx,
        ) = np.shape(dx)
        self.nx = nx
        self.ny = ny

        # Grid spacing
        if hasattr(dx, "__len__"):
            self.dx = dx[int(ny / 2), int(nx / 2)].astype(self.dtype)
            self.dy = dy[int(ny / 2), int(nx / 2)].astype(self.dtype)
        else:
            self.dx = dx
            self.dy = dy

        # Time step
        self.dt = dt

        # Gravity
        self.g = g

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = f.astype(self.dtype)  # full 2D array (needed for sf h<->phi conversion)
            self.f0 = np.nanmean(f).astype(self.dtype)  # scalar reference value for QG operators
            # Beta plane
            self.beta = (f[3:-1, 2:-2] - f[1:-3, 2:-2]) / (2 * self.dy)
        else:
            self.f = np.array(f, dtype=self.dtype)
            self.f0 = np.array(f, dtype=self.dtype)
            self.beta = None

        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = np.nanmean(c).astype(self.dtype)
        else:
            self.c = c.astype(self.dtype)

        # Diagnostics
        _c0 = float(self.c)
        _f0 = float(self.f0)
        _Rd = _c0 / abs(_f0) / 1e3
        print(f"""  - f\u2080 (s\u207b\u00b9):             {_f0:.4e}
  - c  (m/s):              {_c0:.4f}
  - Rd = c/|f\u2080| (km):   {_Rd:.1f}
""")

        # MDT
        self.mdt = mdt

        # Bathymetry
        self.bathymetry_PV_term = bathymetry_PV_term

        # Spatial scheme
        self.upwind = upwind

        # Time scheme
        self.time_scheme = time_scheme

        # Elliptical inversion operator
        x, y = np.meshgrid(np.arange(1, nx - 1, dtype=self.dtype), np.arange(1, ny - 1, dtype=self.dtype))
        laplace_dst = 2 * (np.cos(np.pi / (nx - 1) * x) - 1) / self.dx**2 + 2 * (np.cos(np.pi / (ny - 1) * y) - 1) / self.dy**2
        self.laplace_dst = jnp.asarray(laplace_dst)  # JAX array: no host→device copy in step()
        if self.formulation == "sf":
            self.helmoltz_dst = self.laplace_dst - (self.f0 / self.c) ** 2
        else:
            self.helmoltz_dst = self.g / self.f0 * self.laplace_dst - self.g * self.f0 / self.c**2

        ################
        # Mask array
        ################
        # mask=3 away from the coasts
        mask = np.full((ny, nx), 3, dtype="int8")

        # mask=1 for borders of the domain
        mask[0, :] = 1
        mask[:, 0] = 1
        mask[-1, :] = 1
        mask[:, -1] = 1

        # mask=2 for pixels adjacent to the borders
        mask[1, 1:-1] = 2
        mask[1:-1, 1] = 2
        mask[-2, 1:-1] = 2
        mask[1:-1, -2] = 2

        # mask=0 on land
        if SSH is not None:
            isNAN = np.isnan(SSH)  # get land pixels
            mask[isNAN] = 0
            indNan = np.argwhere(isNAN)
            for i, j in indNan:
                for p1 in range(-2, 3):
                    for p2 in range(-2, 3):
                        itest = i + p1
                        jtest = j + p2
                        if (itest >= 0) & (itest <= ny - 1) & (jtest >= 0) & (jtest <= nx - 1):
                            # mask=1 for coast pixels
                            if (mask[itest, jtest] >= 2) and (p1 in [-1, 0, 1] and p2 in [-1, 0, 1]):
                                mask[itest, jtest] = 1
                            # mask=1 for pixels adjacent to the coast
                            elif mask[itest, jtest] == 3:
                                mask[itest, jtest] = 2

        self.mask = mask
        self.ind0 = mask == 0
        self.ind1 = mask == 1
        self.ind2 = mask == 2
        self.border2 = self.ind1 + self.ind2
        self.ind012 = (self.ind0 + self.ind1 + self.ind2)[2:-2, 2:-2]

        # Diffusion coefficients
        self.Kdiffus = Kdiffus
        self.Kdiffus_trac = Kdiffus_trac

        # Tracer/open-boundary options
        if Wbc is None or np.all(Wbc == 0.0):
            self.Wbc = self.ind1.astype(self.dtype)
        else:
            self.Wbc = Wbc.astype(self.dtype)
        self.bc_trac = bc_trac
        self.sponge_coef = float(sponge_coef)
        self.ocean_h = (self.ind0 == False).astype(self.dtype)

        # Dynamics options
        self.advect_pv = advect_pv
        self.ageo_velocities = ageo_velocities

        # Remember compile flag so _rebuild_helmoltz_dst can re-register JITs
        self._compile = compile

        # JIT compiling functions
        if compile:
            self.h2uv_jit = jit(self.h2uv)
            self.h2pv_jit = jit(self.h2pv)
            self.pv2h_jit = jit(self.pv2h)
            self.rhs_jit = jit(self.rhs)
            # self.adv_jit = jit(self.adv)
            # self.euler_jit = jit(self.euler)
            self.rk2_jit = jit(self.rk2)
            # self.rk3_jit = jit(self.rk3)
            self.bc_jit = jit(self.bc)
            self.one_step_jit = jit(self.one_step)
            self.one_step_for_scan_jit = jit(self.one_step_for_scan)
            self.step_jit = jit(self.step, static_argnums=2)

            self.step_tgl_jit = jit(self.step_tgl, static_argnums=3)
            self.step_adj_jit = jit(self.step_adj, static_argnums=3)

    def _rebuild_helmoltz_dst(self):
        """Recompute helmoltz_dst from the current self.c scalar.
        Both laplace_dst and helmoltz_dst are JAX arrays, so this is a pure
        on-device computation with no host→device copies.
        No JIT re-registration is needed: helmoltz_dst is threaded as an
        explicit JAX array argument through one_step → one_step_for_scan.
        """
        if self.formulation == "sf":
            self.helmoltz_dst = self.laplace_dst - (self.f0 / self.c) ** 2
        else:
            self.helmoltz_dst = self.g / self.f0 * self.laplace_dst - self.g * self.f0 / self.c**2

    def h2uv(self, h):
        """SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity

        """
        if self.formulation == "sf":
            phi = (self.g / self.f) * h
            vk = -jnp.array([[-1, 0, 1], [-1, 0, 1]]) / (4 * self.dx)
            hk = jnp.array([[-1, -1], [0, 0], [1, 1]]) / (4 * self.dy)

            u_ = jsp.signal.convolve(phi, hk, mode="same")
            u = jnp.where(jnp.isnan(u_), 0, u_)
            u = u.at[0].set(0)
            u = u.at[-1].set(0)
            u = u.at[:, 0].set(0)
            # u_ = self.conv(phi, hk)
            # u = u.at[1:-1, 1:].set(jnp.where(jnp.isnan(u_), 0, u_))
            # dv = jnp.roll(phi, 1,1) - jnp.roll(phi, -1,1)
            # v_ = (dv + jnp.roll(dv,1,0)) / (4 * self.dx)
            v_ = jsp.signal.convolve(phi, vk, mode="same")
            v = jnp.where(jnp.isnan(v_), 0, v_)
            v = v.at[0].set(0)
            v = v.at[:, -1].set(0)
            v = v.at[:, 0].set(0)
            # v_ = self.conv(phi, vk)
            # v = v.at[1:, 1:-1].set(jnp.where(jnp.isnan(v_), 0, v_))
        else:
            # TODO : must be replace by conv
            u = u.at[1:-1, 1:].set(-self.g / self.f0 * (h[2:, :-1] + h[2:, 1:] - h[:-2, 1:] - h[:-2, :-1]) / (4 * self.dy))
            v = v.at[1:, 1:-1].set(self.g / self.f0 * (h[1:, 2:] + h[:-1, 2:] - h[:-1, :-2] - h[1:, :-2]) / (4 * self.dx))

            u = jnp.where(jnp.isnan(u), 0, u)
            v = jnp.where(jnp.isnan(v), 0, v)

        return u, v

    def h2pv(self, h, hb, c=None):
        """SSH to PV

        Args:
            h (2D array): SSH field.
            hb (2D array): Background SSH field

        Returns:
            q: Potential Vorticity field
        """

        if c is None:
            c = self.c

        # When c is a 2-D field, extract the interior and boundary sub-arrays
        # so that all index-expressions stay shape-consistent with the sliced
        # phi / h arrays used below.
        if hasattr(c, "shape") and jnp.ndim(c) == 2:
            c_int = c[2:-2, 2:-2]  # (ny-2, nx-2) — for interior update
            c_bdy = c[self.border2]  # flattened boundary values
        else:
            c_int = c
            c_bdy = c
        # Replace NaN (land points) with 0 *before* any arithmetic so that the
        # Laplacian never produces NaN.  The naive pattern
        #   q = jnp.where(jnp.isnan(q), 0, q)
        # masks forward NaN correctly but still lets JAX evaluate
        #   d/dc[-g*f0/c² * h_land] = 2*g*f0/c³ * NaN
        # in the backward pass, giving  0 * NaN = NaN  in the gradient w.r.t. c.
        # Using a safe h here breaks that chain entirely.
        if self.formulation == "sf":
            g_over_f = self.g / self.f
            h = h[1:-1, 1:-1]
            h_safe = jnp.where(jnp.isnan(h), 0.0, h)
            phi = (g_over_f[1:-1, 1:-1]) * h_safe
            phic = phi[1:-1, 1:-1]
            q = jnp.empty((self.ny, self.nx), dtype=self.dtype)
            q = q.at[2:-2, 2:-2].set(
                (jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 0) - 2 * phi)[1:-1, 1:-1] / self.dy**2
                + (phi[1:-1, 2:] + phi[1:-1, :-2] - 2 * phic) / self.dx**2
                - (self.f0 / c_int) ** 2 * phic
            )
            phib = (g_over_f[self.border2]) * hb[self.border2]
            q = q.at[self.border2].set(-((self.f0 / c_bdy) ** 2) * phib)
        else:
            h_safe = jnp.where(jnp.isnan(h), 0.0, h)
            q = jnp.empty((self.ny, self.nx), dtype=self.dtype)
            q = q.at[1:-1, 1:-1].set(
                self.g
                / self.f0
                * (
                    (h_safe[2:, 1:-1] + h_safe[:-2, 1:-1] - 2 * h_safe[1:-1, 1:-1]) / self.dy**2
                    + (h_safe[1:-1, 2:] + h_safe[1:-1, :-2] - 2 * h_safe[1:-1, 1:-1]) / self.dx**2
                )
                - self.g * self.f0 / (c_int**2) * h_safe[1:-1, 1:-1]
            )
            q = q.at[self.border2].set(-self.g * self.f0 / (c_bdy**2) * hb[self.border2])

        q = q.at[self.ind0].set(0)

        return q

    def pv2h(self, q, hb, qb, helmoltz_dst):
        """PV to SSH

        Args:
            q (2D array): SSH field.
            hb (2D array): Background SSH field
            qb (2D array): Background PV field
            helmoltz_dst: DST Helmholtz operator (explicit arg so that
                changing self.helmoltz_dst never requires JIT re-registration).

        Returns:
            h: SSH field
        """

        # Interior pv
        qin = (q - qb)[1:-1, 1:-1]
        inv = inverse_elliptic_dst(qin, helmoltz_dst)
        h = hb
        if self.formulation == "sf":
            h = h.at[1:-1, 1:-1].add(self.f[1:-1, 1:-1] / self.g * inv)
        else:
            h = h.at[1:-1, 1:-1].add(inv)
        return h

    def rhs(self, u, v, ua, va, var0, way=1):
        """increment right hand side

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment

        """

        if len(var0.shape) == 3:
            q0 = var0[0]
            c0 = var0[1:]
        else:
            q0 = var0
            c0 = None

        # FIXME work with empty but not sure why
        incr = jnp.zeros_like(var0, dtype=self.dtype)
        # incr = jnp.empty_like(var0, dtype=self.dtype)

        #######################
        # Upwind current
        #######################
        u_on_T = way * 0.5 * (u + jnp.roll(u, -1, 1))[2:-2, 2:-2]
        v_on_T = way * 0.5 * (v + jnp.roll(v, -1, 0))[2:-2, 2:-2]
        mu = u_on_T < 0
        mv = v_on_T < 0
        if self.ageo_velocities:
            up = jnp.where(mu, 0, u_on_T)
            um = jnp.where(~mu, 0, u_on_T)
            vp = jnp.where(mv, 0, v_on_T)
            vm = jnp.where(~mv, 0, v_on_T)

        # PV advection
        f_to_adv = 0
        f_to_adv_set = False
        if self.advect_pv:
            f_to_adv = q0
            f_to_adv_set = True
        if self.bathymetry_PV_term is not None:
            f_to_adv += (self.f if self.formulation == "sf" else self.f0) * self.bathymetry_PV_term
            f_to_adv_set = True
        if not f_to_adv_set:
            rhs_q = jnp.zeros_like(up, dtype=self.dtype)
        else:
            # allow a single call to adv
            rhs_q = self.advm(u_on_T, v_on_T, mu, mv, f_to_adv)

        # Beta plane
        if self.beta is not None:
            rhs_q -= self.beta * v_on_T

        # PV Diffusion
        if self.Kdiffus is not None:
            rhs_q += self.Kdiffus * diffusion(q0, self.dx, self.dy)

        rhs_q = jnp.where(jnp.logical_or(jnp.isnan(rhs_q), self.ind012), 0, rhs_q)

        if c0 is not None:
            incr = incr.at[0, 2:-2, 2:-2].set(rhs_q)

            if self.ageo_velocities:
                ua_on_T = way * 0.5 * (ua[2:-2, 2:-2] + ua[2:-2, 3:-1])
                va_on_T = way * 0.5 * (va[2:-2, 2:-2] + va[3:-1, 2:-2])
                mua = ua_on_T < 0
                mva = va_on_T < 0
                up += jnp.where(mua, 0, ua_on_T)
                um += jnp.where(~mua, 0, ua_on_T)
                vp += jnp.where(mva, 0, va_on_T)
                vm += jnp.where(~mva, 0, va_on_T)

                # FIXME : shifted ? mask is needed only if ageo_velocities is True?
                u_mask = self.ocean_h[2:-2, 2:-2]
                up = up * u_mask
                um = um * u_mask
                vp = vp * u_mask
                vm = vm * u_mask

            for i, c0_ in enumerate(c0):
                if self.ageo_velocities:
                    rhs_c = self.adv(up, vp, um, vm, c0_)
                else:
                    rhs_c = self.advm(u_on_T, v_on_T, mu, mv, c0_)
                if self.Kdiffus_trac is not None:
                    rhs_c += self.Kdiffus_trac * diffusion(c0_, self.dx, self.dy)
                rhs_c = jnp.where(jnp.logical_or(jnp.isnan(rhs_c), self.ind0[2:-2, 2:-2]), 0, rhs_c)
                incr = incr.at[i + 1, 2:-2, 2:-2].set(rhs_c)
        else:
            incr = incr.at[2:-2, 2:-2].set(rhs_q)

        return incr

    def adv(self, up, vp, um, vm, q0):
        """
        3rd-order upwind scheme
        """
        q0_2_0 = 3 * q0[2:-2, 2:-2]
        q0_2_1Y = 2 * q0[2:-2, 3:-1]
        q0_2__1Y = 2 * q0[2:-2, 1:-3]
        q0_2_1X = 2 * q0[3:-1, 2:-2]
        q0_2__1X = 2 * q0[1:-3, 2:-2]

        ugradq = (
            -up / (6 * self.dx) * (q0_2_1Y + q0_2_0 - 3 * q0_2__1Y + q0[2:-2, :-4])
            + um / (6 * self.dx) * (q0_2__1Y + q0_2_0 - 3 * q0_2_1Y + q0[2:-2, 4:])
            - vp / (6 * self.dy) * (q0_2_1X + q0_2_0 - 3 * q0_2__1X + q0[:-4, 2:-2])
            + vm / (6 * self.dy) * (q0_2__1X + q0_2_0 - 3 * q0_2_1X + q0[4:, 2:-2])
        )

        return ugradq

    @staticmethod
    def conv(q, k):
        return jsp.signal.convolve(q, k, mode="valid")

    @staticmethod
    def hconv(q, k, N=2):
        return jsp.signal.convolve(q[N:-N], k, mode="valid")

    @staticmethod
    def vconv(q, k, N=2):
        return jsp.signal.convolve(q[:, N:-N], k, mode="valid")

    def advm(self, u, v, mu, mv, q0):
        """
        3rd-order upwind scheme
        """
        umk = jnp.array([[1, -6, 3, 2, 0]]) / (6 * self.dx)
        upk = -jnp.array([[0, 2, 3, -6, 1]]) / (6 * self.dx)
        vmk = jnp.array([[1], [-6], [3], [2], [0]]) / (6 * self.dy)
        vpk = -jnp.array([[0], [2], [3], [-6], [1]]) / (6 * self.dy)

        ugradq = jnp.where(mu, self.hconv(q0, umk), self.hconv(q0, upk)) * u + jnp.where(mv, self.vconv(q0, vmk), self.vconv(q0, vpk)) * v
        return ugradq

    def euler(self, var0, incr, way):
        """
        Euler time scheme
        """

        return var0 + way * self.dt * incr

    def rk2(self, var0, incr, ua, va, hb, qb, way, helmoltz_dst):
        """
        2rd-order Runge-Kutta time scheme
        """

        # k2
        var12 = var0 + 0.5 * self.dt * incr
        if len(incr.shape) == 3:
            q12 = var12[0]
        else:
            q12 = +var12
        h12 = self.pv2h_jit(q12, hb, qb, helmoltz_dst)
        u12, v12 = self.h2uv_jit(h12)
        if len(incr.shape) == 3:
            var12 = var12.at[0].set(q12)
        else:
            var12 = q12
        incr12 = self.rhs_jit(u12, v12, ua, va, var12, way=way)

        return var0 + self.dt * incr12

    def rk3(self, var0, incr, ua, va, hb, qb, way, helmoltz_dst):
        """
        3rd-order Runge-Kutta time scheme
        """
        # k1
        var1 = var0 + self.dt * incr

        # k2
        var12 = var0 + 0.5 * self.dt * incr
        if len(incr.shape) == 3:
            q12 = var12[0]
        else:
            q12 = +var12
        h12 = self.pv2h_jit(q12, hb, qb, helmoltz_dst)
        u12, v12 = self.h2uv_jit(h12)
        if len(incr.shape) == 3:
            var12 = var12.at[0].set(q12)
        else:
            var12 = +q12
        incr12 = self.rhs_jit(u12, v12, ua, va, var12, way=way)

        # k3
        var13 = var0 - self.dt * incr + 2 * self.dt * incr12
        if len(incr.shape) == 3:
            q13 = var13[0]
        else:
            q13 = +var13
        h13 = self.pv2h_jit(q13, hb, qb, helmoltz_dst)
        u13, v13 = self.h2uv_jit(h13)
        if len(incr.shape) == 3:
            var13 = var13.at[0].set(q13)
        else:
            var13 = +q13
        incr13 = self.rhs_jit(u13, v13, ua, va, var13, way=way)

        return var0 + (self.dt / 6) * (incr + 4 * incr12 + incr13)

    def bc(self, var1, var0, u, v, varb):
        """
        Open Boundary Conditions for tracers, following Mellor (1996)
        https://www.researchgate.net/publication/242777179_Users_Guide_For_A_Three-Dimensional_Primitive_Equation_Numerical_Ocean_Model
        """
        if len(varb.shape) == 3 and varb.shape[0] > 1:
            if self.bc_trac == "OBC":
                v_sborder = abs(v[1, 1:-1]) * self.dt / self.dy
                v_nborder = abs(v[-1, 1:-1]) * self.dt / self.dy
                v_wborder = abs(u[1:-1, 1]) * self.dt / self.dx
                v_eborder = abs(u[1:-1, -1]) * self.dt / self.dx
                m_sborder = v[1, 1:-1] < 0
                m_nborder = v[-1, 1:-1] < 0
                m_wborder = u[1:-1, 1] < 0
                m_eborder = u[1:-1, -1] < 0

            for i_, (var0_, varb_) in enumerate(zip(var0[1:], varb[1:])):
                i = i_ + 1
                if self.bc_trac == "OBC":
                    s0, s1 = var0_[:2, 1:-1]
                    var1 = var1.at[i, 0, 1:-1].set(s0 * (1 - v_sborder) + jnp.where(m_sborder, s1, varb_[0, 1:-1]) * v_sborder)
                    n1, n0 = var0_[-2:, 1:-1]
                    var1 = var1.at[i, -1, 1:-1].set(n0 * (1 - v_nborder) + jnp.where(m_nborder, varb_[-1, 1:-1], n1) * v_nborder)
                    w0, w1 = var0_[1:-1, :2].T
                    var1 = var1.at[i, 1:-1, 0].set(w0 * (1 - v_wborder) + jnp.where(m_wborder, w1, varb_[1:-1, 0]) * v_wborder)
                    e1, e0 = var0_[1:-1, -2:].T
                    var1 = var1.at[i, 1:-1, -1].set(e0 * (1 - v_eborder) + jnp.where(m_eborder, varb_[1:-1, -1], e1) * v_eborder)
                else:
                    var1 = var1.at[i, self.border2].set(varb[i, self.border2])

                if self.sponge_coef > 0.0:
                    var1 = var1.at[i, 1:-1, 1:-1].add(
                        self.sponge_coef * self.Wbc[1:-1, 1:-1] * self.ocean_h[1:-1, 1:-1] * (varb_[1:-1, 1:-1] - var1[i, 1:-1, 1:-1])
                    )

                var1 = var1.at[i, self.ind0].set(varb[i, self.ind0])

        return var1

    def one_step(self, h0, ua, va, var0, hb, varb, way=1, helmoltz_dst=None):
        """
        One step forward
        """

        if helmoltz_dst is None:
            helmoltz_dst = jnp.asarray(self.helmoltz_dst)

        # Compute geostrophic velocities
        u, v = self.h2uv_jit(h0)

        # Boundary field for PV
        if len(varb.shape) == 3:
            qb = +varb[0]
        else:
            qb = +varb

        # Compute increment
        incr = self.rhs_jit(u, v, ua, va, var0, way=way)
        # Time integration
        if self.time_scheme == "Euler":
            var1 = self.euler_jit(var0, incr, way)
        elif self.time_scheme == "rk2":
            var1 = self.rk2_jit(var0, incr, ua, va, hb, qb, way, helmoltz_dst)
        elif self.time_scheme == "rk3":
            var1 = self.rk3_jit(var0, incr, ua, va, hb, qb, way, helmoltz_dst)
        else:
            raise ValueError(f"Unsupported time_scheme: {self.time_scheme}")

        # Elliptical inversion — uses the passed helmoltz_dst so that a
        # changed self.helmoltz_dst is picked up without JIT re-registration.
        if len(var1.shape) == 3:
            q1 = +var1[0]
        else:
            q1 = +var1
        h1 = self.pv2h_jit(q1, hb, qb, helmoltz_dst)

        var1 = self.bc_jit(var1, var0, u + ua, v + va, varb)

        return h1, var1

    def one_step_for_scan(self, X0, X):
        """
        One step forward for scan.  helmoltz_dst is carried as a
        JAX array so the scan kernel never needs recompilation when
        self.helmoltz_dst is updated between optimizer iterations.
        """

        h1, ua, va, var1, hb, varb, helmoltz_dst = X0
        h1, var1 = self.one_step_jit(h1, ua, va, var1, hb, varb, helmoltz_dst=helmoltz_dst)
        X = (h1, ua, va, var1, hb, varb, helmoltz_dst)
        # method need to argument return
        return X, None

    def step(self, X0, Xb, nstep=1, c=None):
        """Propagation

        Args:
            X0 (2D or 3D array): initial SSH or stacked state
            Xb (2D or 3D array): boundary SSH or stacked boundaries
            nstep (int): number of time-step
            c (2D array or None): effective phase-speed field c_eff(x,y).
                When None (default) the prior self.c scalar baked at __init__
                is used — behaviour is bit-for-bit identical to the old code.
                When provided, an area-weighted ocean-mean scalar c_bar is
                derived and the DST operator is recomputed from c_bar; the
                full 2-D c field is passed to h2pv so the stretching term
                uses the spatially-varying c_eff.

        Returns:
            X1 (2D or 3D array): propagated state

        """
        ua0 = va0 = None
        if len(X0.shape) == 3:
            h0 = X0[0]
            if self.ageo_velocities:
                ua0 = X0[1]
                va0 = X0[2]
                c0 = X0[3:]
            else:
                c0 = X0[1:]
            hb = Xb[0]
            cb = Xb[1:]
        else:
            h0 = X0
            c0 = None
            hb = Xb
            cb = None

        if c0 is not None:
            assert cb is not None
            c0 = c0.at[:, self.ind0].set(cb[:, self.ind0])

        if self.mdt is not None:
            h0 += self.mdt
            hb += self.mdt

        if c is None:
            # ---- fast path: prior c scalar, baked DST operator (today's code) ----
            q0 = self.h2pv_jit(h0, hb)
            qb = self.h2pv_jit(hb, hb)

            h1 = h0
            var1 = q0
            varb = qb
            if self.ageo_velocities:
                assert ua0 is not None and va0 is not None
                ua = ua0
                va = va0
            else:
                ua = jnp.zeros_like(h0)
                va = jnp.zeros_like(h0)
            if c0 is not None:
                var1 = jnp.append(var1[jnp.newaxis, :, :], c0, axis=0)
                varb = jnp.append(varb[jnp.newaxis, :, :], cb, axis=0)

            helmoltz = jnp.asarray(self.helmoltz_dst)
            X1, _ = scan(self.one_step_for_scan_jit, init=(h1, ua, va, var1, hb, varb, helmoltz), length=nstep)
            h1, ua, va, var1, hb, varb, _ = X1

        else:
            # ---- c-aware path: c_eff(x,y) only enters h2pv ----
            q0 = self.h2pv_jit(h0, hb, c)
            qb = self.h2pv_jit(hb, hb, c)

            h1 = h0
            var1 = q0
            varb = qb
            if self.ageo_velocities:
                assert ua0 is not None and va0 is not None
                ua = ua0
                va = va0
            else:
                ua = jnp.zeros_like(h0)
                va = jnp.zeros_like(h0)
            if c0 is not None:
                var1 = jnp.append(var1[jnp.newaxis, :, :], c0, axis=0)
                varb = jnp.append(varb[jnp.newaxis, :, :], cb, axis=0)

            helmoltz = jnp.asarray(self.helmoltz_dst)
            X1, _ = scan(self.one_step_for_scan_jit, init=(h1, ua, va, var1, hb, varb, helmoltz), length=nstep)
            h1, ua, va, var1, hb, varb, _ = X1

        # Mask
        h1 = h1.at[self.ind0].set(jnp.nan)

        if len(var1.shape) == 3:
            var1 = var1.at[1:, self.ind0].set(np.nan)

        if self.mdt is not None:
            h1 -= self.mdt

        if len(var1.shape) == 3:
            if self.ageo_velocities:
                assert ua0 is not None and va0 is not None
                X1 = jnp.concatenate((h1[jnp.newaxis, :, :], ua0[jnp.newaxis, :, :], va0[jnp.newaxis, :, :], var1[1:]), axis=0)
            else:
                X1 = var1.at[0].set(h1)
        else:
            X1 = +h1

        return X1

    def step_tgl(self, dX0, X0, Xb, nstep=1, c=None):

        _, dX1 = jvp(partial(self.step_jit, Xb=Xb, nstep=nstep, c=c), (X0,), (dX0,))

        return dX1

    def step_adj(self, adX0, X0, Xb, nstep=1, c=None):
        _, adf = vjp(partial(self.step_jit, Xb=Xb, nstep=nstep, c=c), X0)
        return adf(adX0)[0]


if __name__ == "__main__":

    ny, nx = 100, 100
    dx = 10e3 * jnp.ones((ny, nx))
    dy = 12e3 * jnp.ones((ny, nx))
    dt = 300

    SSH0 = np.random.random((ny, nx)).astype("float64")
    c = 2.7 * np.ones((ny, nx), dtype="float64")
    f = 1e-4 * np.ones((ny, nx), dtype="float64")

    qgm = Qgm(dx=dx, dy=dy, dt=dt, c=c, f=f, SSH=SSH0)

    SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

    ####################
    # h2pv
    ####################
    if False:
        print("*** h2pv ***")
        # Current trajectory
        SSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")
        SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        # Perturbation
        dSSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        # Adjoint
        adPV = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        def h2pv_tgl(dh0, h0, hb):

            _, dh1 = jvp(partial(qgm.h2pv, hbc=hb, ib=0), (h0,), (dh0,))

            return dh1

        def h2pv_adj(adq0, h0, hb):

            _, adf = vjp(partial(qgm.h2pv, hbc=hb, ib=0), h0)

            return adf(adq0)[0]

        # Forward
        PV0 = qgm.h2pv(SSH, SSHb, ib=0).astype("float64")

        print("Tangent test:")
        for p in range(10):
            lambd = 10 ** (-p)

            PV1 = qgm.h2pv(SSH + lambd * dSSH, SSHb, ib=0).astype("float64")
            dPV = h2pv_tgl(lambd * dSSH, SSH, SSHb).astype("float64")

            ps = jnp.linalg.norm((PV1 - PV0 - dPV).flatten()) / jnp.linalg.norm(dPV)

            print("%.E" % lambd, "%.E" % ps)

        # Adjoint test
        dPV = h2pv_tgl(dSSH, SSH, SSHb).astype("float64")
        adSSH = h2pv_adj(adPV, SSH, SSHb).astype("float64")

        ps1 = jnp.inner(dPV.flatten(), adPV.flatten())
        ps2 = jnp.inner(dSSH.flatten(), adSSH.flatten())

        print("\nAdjoint test:", ps1 / ps2)

    ####################
    # pv2h
    ####################
    if False:
        print("*** pv2h ***")
        # Current trajectory
        PV = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")
        SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")
        PVb = qgm.h2pv(SSHb, SSHb, ib=0).astype("float64")

        # Perturbation
        dPV = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        # Adjoint
        adSSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        def pv2h_tgl(dq, q, hb, qb):

            _, dh = jvp(partial(qgm.pv2h, hb=hb, qb=qb, ib=0), (q,), (dq,))

            return dh

        def pv2h_adj(adh, q, hb, qb):

            _, adf = vjp(partial(qgm.pv2h, hb=hb, qb=qb, ib=0), q)

            return adf(adh)[0]

        # Forward
        SSH = qgm.pv2h(PV, SSHb, PVb, ib=0).astype("float64")

        print("Tangent test:")
        for p in range(10):
            lambd = 10 ** (-p)

            SSH1 = qgm.pv2h(PV + lambd * dPV, SSHb, PVb, ib=0).astype("float64")
            dSSH = pv2h_tgl(lambd * dPV, PV, SSHb, PVb).astype("float64")

            ps = jnp.linalg.norm((SSH1 - SSH - dSSH).flatten()) / jnp.linalg.norm(dSSH)

            print("%.E" % lambd, "%.E" % ps)

        # Adjoint test
        dSSH = pv2h_tgl(dPV, PV, SSHb, PVb).astype("float64")
        adPV = pv2h_adj(adSSH, PV, SSHb, PVb).astype("float64")

        ps1 = jnp.inner(dPV.flatten(), adPV.flatten())
        ps2 = jnp.inner(dSSH.flatten(), adSSH.flatten())

        print("\nAdjoint test:", ps1 / ps2)

    ####################
    # step
    ####################
    if False:
        print("*** step ***")
        # Current trajectory
        SSH0 = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")
        SSHb = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        # Perturbation
        dSSH = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        # Adjoint
        adSSH0 = jnp.array(1e-2 * np.random.random((ny, nx))).astype("float64")

        # Tangent test
        SSH2 = qgm.step_jit(X0=SSH0, Xb=SSHb).astype("float64")
        print("Tangent test:")
        for p in range(10):
            lambd = 10 ** (-p)

            SSH1 = qgm.step_jit(X0=SSH0 + lambd * dSSH, Xb=SSHb).astype("float64")
            dSSH1 = qgm.step_tgl_jit(dX0=lambd * dSSH, X0=SSH0, Xb=SSHb).astype("float64")

            mask = jnp.isnan(SSH1 - SSH2 - dSSH1)
            ps = jnp.linalg.norm((SSH1 - SSH2 - dSSH1)[~mask].flatten()) / jnp.linalg.norm(dSSH1[~mask])

            print("%.E" % lambd, "%.E" % ps)

        # Adjoint test
        dSSH1 = qgm.step_tgl_jit(dX0=dSSH, X0=SSH0, Xb=SSHb).astype("float64")
        adSSH1 = qgm.step_adj_jit(adX0=adSSH0, X0=SSH0, Xb=SSHb).astype("float64")

        mask = jnp.isnan(dSSH1 + adSSH1 + SSH0 + dSSH)

        ps1 = jnp.inner(dSSH1[~mask].flatten(), adSSH0[~mask].flatten())
        ps2 = jnp.inner(dSSH[~mask].flatten(), adSSH1[~mask].flatten())

        print("\nAdjoint test:", ps1 / ps2)
