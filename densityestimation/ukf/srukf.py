# Origin: UKF.m (Mehta 2018 base; Gondelach 2020; Li 2022)
# License: GNU GPL v3
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from densityestimation.ukf.unscented import unscented_transform


def _cholesky_psd(M: np.ndarray) -> np.ndarray:
    """
    対称正定/半正定を想定。コレスキー失敗時は対角上でフォールバック。
    返り値は「下三角」S（S @ S.T ≈ M）
    """
    try:
        return np.linalg.cholesky(M + 0.0 * M.T)  # 対称化のための +0*MT
    except np.linalg.LinAlgError:
        # 対角（たとえば R,Q が対角のとき MATLAB の sqrt(R) と等価）
        d = np.clip(np.diag(M), 0.0, None)
        return np.diag(np.sqrt(d))


def _cholupdate_upper(R: np.ndarray, x: np.ndarray, sign: str = '+') -> np.ndarray:
    """
    Upper三角のRに対し rank-1 の更新/ダウndate（MATLAB cholupdate 相当）。
    R'R (+/-) x x' を満たす上三角を返す。
    """
    R = R.copy()
    x = x.astype(float).copy()
    n = x.size
    for k in range(n):
        rkk = R[k, k]
        xk = x[k]
        if sign == '+':
            r = np.hypot(rkk, xk)
            c = r / rkk if rkk != 0 else 0.0
            s = xk / r if r != 0 else 0.0
        else:
            # downdate
            r = np.sqrt(max(rkk**2 - xk**2, 0.0))
            c = r / rkk if rkk != 0 else 0.0
            s = xk / rkk if rkk != 0 else 0.0
        R[k, k] = r
        if k + 1 < n:
            if sign == '+':
                R[k, k+1:] = (R[k, k+1:] + s * x[k+1:]) / c if c != 0 else 0.0
                x[k+1:] = c * x[k+1:] - s * R[k, k+1:]
            else:
                R[k, k+1:] = (R[k, k+1:] - s * x[k+1:]) / c if c != 0 else 0.0
                x[k+1:] = c * x[k+1:] - s * R[k, k+1:]
    return R


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """角度を [-pi, pi] に正規化（要素配列想定）。"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def ukf(
    X_est: np.ndarray,
    Meas: np.ndarray,
    time: np.ndarray,
    stateFnc: Callable[[np.ndarray, float, float], np.ndarray],
    measurementFcn: Callable[[np.ndarray], np.ndarray],
    P: np.ndarray,
    RM: np.ndarray,
    Q: np.ndarray,
    angle_block: int = 6,
    kappa: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Square-root Unscented Kalman filter（MATLAB UKF.m の忠実移植）
    Parameters
    ----------
    X_est : (n, m) ndarray
        初期列 X_est[:,0] を与え、以降の列をこの関数で更新して返す。
    Meas  : (ny, m) ndarray
        観測ベクトルの系列（列が時刻）。
    time  : (m,) or (m,1) ndarray
        観測時刻（連続 or 等間隔どちらでも）。stateFncにそのまま渡す。
    stateFnc : f(xx, t_i, t_{i+1}) -> (n, 2n+1) ndarray
        シグマ点行列 xx (n,2n+1) を t_i→t_{i+1} に伝播させた Xp を返す関数。
    measurementFcn : g(Xp) -> (ny, 2n+1) ndarray
        伝播済みシグマ点 Xp を観測空間へ写像。
    P : (n,n) ndarray
        初期状態共分散。
    RM : (ny,ny) ndarray
        観測ノイズ共分散。
    Q : (n,n) ndarray
        プロセスノイズ共分散（離散時刻の増分として扱う）。
    angle_block : int
        MEE の L（真近点離角）が 6 番目である前提から、6要素ごとに6番目を wrapToPi。
    kappa : float | None
        Unscented Transform の κ。省略時は MATLAB と同じ κ=3−L。
    Returns
    -------
    X_est : (n,m) ndarray
        推定時系列（列が時刻）。
    Pv : (n,m) ndarray
        各時刻の共分散対角（分散）の時系列。
    """
    X_est = np.array(X_est, dtype=float, copy=True)
    Meas = np.array(Meas, dtype=float, copy=False)
    time = np.array(time, dtype=float).reshape(-1)
    n, m = X_est.shape
    ny = Meas.shape[0]

    # Unscented weights（MATLAB Unscented_Transform と同一仕様: α=1, β=2, κ=3−L 既定）
    Wm, Wc, L, lam = unscented_transform(X_est[:, 0], kappa=kappa)
    Wm0, Wmi = Wm[0], Wm[1]
    Wc0, Wci = Wc[0], Wc[1]
    eta = np.sqrt(L + lam)

    # Square-root factors
    S = _cholesky_psd(P)                  # lower-triangular (S @ S.T = P)
    SR_R = _cholesky_psd(RM)              # measurement noise (lower)
    SR_Q = _cholesky_psd(Q)               # process noise (lower)

    Pv = np.zeros((n, m), dtype=float)

    try:
        for i in range(m - 1):
            # --- Sigma points around current estimate x_i
            sigv = np.hstack([eta * S, -eta * S])        # (n, 2n)
            xx = np.hstack([X_est[:, [i]], X_est[:, [i]] + sigv])  # (n, 2n+1)

            # --- Time update: propagate sigma points
            Xp = stateFnc(xx, time[i], time[i+1])        # (n, 2n+1)

            # Predicted mean
            X_est[:, i+1] = Wm0 * Xp[:, 0] + Wmi * np.sum(Xp[:, 1:], axis=1)

            # --- Get propagated square-root via QR + cholupdate（MATLAB準拠）
            # Build [ sqrt(Wci)*(Xp[:,1:]-x_pred) , SR_Q ]' then thin-QR
            dXp = Xp[:, 1:] - X_est[:, [i+1]]
            A = np.hstack([np.sqrt(Wci) * dXp, SR_Q])    # (n, 2n + nQ)
            # QR of A' gives R upper-tri
            _, R = np.linalg.qr(A.T, mode="reduced")
            S_minus = R.T                                 # lower
            # cholupdate with Wc0*(Xp[:,0]-x_pred)
            S_minus_T = _cholupdate_upper(R, np.sqrt(Wc0) * (Xp[:, 0] - X_est[:, i+1]), sign='+')
            S_minus = S_minus_T.T

            # --- Measurement prediction
            Ym = measurementFcn(Xp)                      # (ny, 2n+1)
            ym = Wm0 * Ym[:, 0] + Wmi * np.sum(Ym[:, 1:], axis=1)

            # Wrap true longitude residuals（6,12,18,...行）
            if angle_block is not None and angle_block > 0 and ym.size >= angle_block:
                idx = slice(angle_block-1, ym.size, angle_block)
                DY0 = Ym[:, 0] - ym
                DY0[idx] = _wrap_to_pi(DY0[idx])
                DY2 = Ym[:, 1:] - ym[:, None]
                DY2[idx, :] = _wrap_to_pi(DY2[idx, :])
            else:
                DY0 = Ym[:, 0] - ym
                DY2 = Ym[:, 1:] - ym[:, None]

            # --- Innovation covariance S_y via QR + cholupdate
            B = np.hstack([np.sqrt(Wci) * DY2, SR_R])    # (ny, 2n + nyR)
            _, Ry = np.linalg.qr(B.T, mode="reduced")    # Ry upper
            Sy = _cholupdate_upper(Ry, np.sqrt(Wc0) * DY0, sign='+').T  # Sy lower

            # --- Cross covariance Pxy
            Pxy0 = Wc0 * np.outer(Xp[:, 0] - X_est[:, i+1], DY0)
            Pmat = Xp[:, 1:] - X_est[:, [i+1]]
            Pyymat = DY2
            Pxy = Pxy0 + Wci * (Pmat @ Pyymat.T)         # (n, ny)

            # --- Residual and gain
            yres = Meas[:, i+1] - ym
            if angle_block is not None and angle_block > 0 and yres.size >= angle_block:
                yres[idx] = _wrap_to_pi(yres[idx])

            # KG = Pxy / Sy' / Sy  （安定に： solve で二段階）
            Z = np.linalg.solve(Sy.T, Pxy.T)
            KG = np.linalg.solve(Sy, Z).T                 # (n, ny)

            # State update
            X_est[:, i+1] = X_est[:, i+1] + KG @ yres

            # Covariance downdate S ← S_minus ⊖ U(:,j) 反復
            U = KG @ Sy                                   # (n, ny)
            Rm = S_minus.T                                # upper
            for j in range(ym.size):
                Rm = _cholupdate_upper(Rm, U[:, j], sign='-')
            S = Rm.T
            Pv[:, i+1] = np.diag(S @ S.T)
    except Exception:
        raise

    return X_est, Pv
