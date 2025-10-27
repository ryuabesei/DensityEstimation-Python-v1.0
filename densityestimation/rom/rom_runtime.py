import numpy as np


class ROMRuntime:
    """
    Reduced-Order Model (ROM) runtime for density estimation.
    Handles state propagation (step_z) and density reconstruction (density_at)
    based on POD/DMDc models.

    Attributes
    ----------
    Ur : ndarray
        POD basis vectors [n_grid, r].
    xbar : ndarray
        Mean log10(density) field [n_grid].
    Ac : ndarray
        Continuous-time state matrix [r, r].
    Bc : ndarray
        Continuous-time input matrix [r, m].
    dt_h : float
        Time step in hours (training grid step, e.g., 1.0).
    Qz : ndarray
        Process noise diagonal elements for z (from 1h forecast error).
    input_fn : callable
        Function of t_epoch returning external forcing vector (space weather indices).
    """

    def __init__(self, Ur, xbar, Ac, Bc, dt_hours: float, Qz_diag, input_fn):
        self.Ur = np.asarray(Ur, float)
        self.xbar = np.asarray(xbar, float)
        self.Ac = np.asarray(Ac, float)
        self.Bc = np.asarray(Bc, float)
        self.dt_h = float(dt_hours)
        self.Qz = np.asarray(Qz_diag, float)
        self.input_fn = input_fn

    # ---- 基本プロパティ ----
    @property
    def r(self) -> int:
        """ROMの次元"""
        return self.Ur.shape[1]

    # ---- 1時間分の状態推移 ----
    def step_z(self, z: np.ndarray, t_epoch: float, dt_sec: float, discretize_lin) -> np.ndarray:
        """
        Advance reduced state z using continuous DMDc matrices (Ac, Bc).

        Parameters
        ----------
        z : ndarray, shape (r,)
            Current reduced state.
        t_epoch : float
            Julian date.
        dt_sec : float
            Time step [s].
        discretize_lin : callable
            Continuous-to-discrete conversion function, e.g., expm-based.

        Returns
        -------
        z_next : ndarray
            Propagated reduced state.
        """
        z = np.asarray(z, float)
        u = np.asarray(self.input_fn(t_epoch), float)
        Ad, Bd = discretize_lin(self.Ac, self.Bc, dt_sec)
        return Ad @ z + Bd @ u

    # ---- 密度再構成 ----
    def density_at(self, z: np.ndarray, r_vec: np.ndarray, t_epoch: float, grid_interp_fn) -> float:
        """
        Reconstruct local density at position r_vec.

        Parameters
        ----------
        z : ndarray
            Current reduced state.
        r_vec : ndarray
            Position vector [km].
        t_epoch : float
            Julian date.
        grid_interp_fn : callable
            Interpolation from reconstructed grid log10(rho) → rho(r_vec).

        Returns
        -------
        rho : float
            Atmospheric density [kg/m^3] (or model’s native unit).
        """
        log10rho_grid = self.xbar + self.Ur @ np.asarray(z, float)
        rho = grid_interp_fn(log10rho_grid, r_vec, t_epoch)
        return float(rho)
