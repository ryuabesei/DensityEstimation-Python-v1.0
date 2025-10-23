# densityestimation/Astrofunction/gravity_model.py
# License: GNU GPL v3
#重力ポテンシャルと加速度（J2〜Jn、EGM2008のLnmなど）計算
#EGM2008 等の正規化球面調和係数 (Cnm, Snm) を使って、ECEF 座標系での地球重力加速度 [m/s²] を計算

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ==============================
# 1) ECEF 重力加速度（球面調和展開）
# ==============================

def compute_earth_gravitational_acceleration(
    rr_ecef: np.ndarray,
    GM: float,
    Re: float,
    C: np.ndarray,
    S: np.ndarray,
    gravdegree: int,
    scale_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Earth gravitational acceleration in ECEF coordinates
    using spherical harmonic expansion.

    Parameters
    ----------
    rr_ecef : ndarray, shape (N, 3)
        Position vectors in ECEF [m]
    GM : float
        Earth gravitational parameter [m^3/s^2]
    Re : float
        Earth mean equatorial radius [m]
    C, S : ndarray
        Normalized spherical harmonic coefficients (Cnm, Snm),
        shape >= (gravdegree+3, gravdegree+3)
    gravdegree : int
        Maximum degree/order of gravity model (e.g., 48)
    scale_factor : ndarray
        Normalization scale factors used in derivative recursion,
        shape = (gravdegree+3, gravdegree+3)

    Returns
    -------
    ax, ay, az : ndarray
        Gravitational acceleration components [m/s^2]
    """
    if rr_ecef.ndim != 2 or rr_ecef.shape[1] != 3:
        raise ValueError("Input rr_ecef must have shape (N, 3).")

    N = rr_ecef.shape[0]
    r = np.linalg.norm(rr_ecef, axis=1)
    if np.any(r < Re):
        raise ValueError("Radial position below Earth's surface detected.")

    # geocentric latitude
    phic = np.arcsin(rr_ecef[:, 2] / r)
    # longitude
    lam = np.arctan2(rr_ecef[:, 1], rr_ecef[:, 0])

    # cos(m*lambda), sin(m*lambda) を漸化式で
    smlambda = np.zeros((N, gravdegree + 1))
    cmlambda = np.zeros((N, gravdegree + 1))
    sl = np.sin(lam)
    cl = np.cos(lam)
    # m=0
    smlambda[:, 0] = 0.0
    cmlambda[:, 0] = 1.0
    if gravdegree >= 1:
        # m=1
        smlambda[:, 1] = sl
        cmlambda[:, 1] = cl
    for m in range(2, gravdegree + 1):
        # cos/sin の重倍角漸化式
        smlambda[:, m] = 2.0 * cl * smlambda[:, m - 1] - smlambda[:, m - 2]
        cmlambda[:, m] = 2.0 * cl * cmlambda[:, m - 1] - cmlambda[:, m - 2]

    # 正規化付きルジャンドル陪多項式
    P = _compute_legendre_polynomials(phic, gravdegree)

    ax, ay, az = _compute_gravity(
        rr_ecef, gravdegree, P, C, S, smlambda, cmlambda, GM, Re, r, scale_factor
    )
    return ax, ay, az


def _compute_legendre_polynomials(phi: np.ndarray, maxdeg: int) -> np.ndarray:
    """
    MATLAB computeLegendrePolynomials と同等のインデックスに合わせる。
    ここでは 0-based で [n+1, m] に相当する場所を使う実装にしてある。

    P の shape は (maxdeg+3, maxdeg+3, N)
    """
    N = len(phi)
    P = np.zeros((maxdeg + 3, maxdeg + 3, N), dtype=float)

    # 極角（共緯度）の cos/sin
    cphi = np.cos(np.pi / 2.0 - phi)
    sphi = np.sin(np.pi / 2.0 - phi)

    # Seeds（MATLAB: P(1,1)=1, P(2,1)=sqrt(3)cphi, P(2,2)=sqrt(3)sphi）
    # 0-based に直すと:
    #   P[0,0]=1
    #   P[1,0]=sqrt(3)*cphi
    #   P[1,1]=sqrt(3)*sphi
    P[0, 0, :] = 1.0
    P[1, 0, :] = np.sqrt(3.0) * cphi
    P[1, 1, :] = np.sqrt(3.0) * sphi

    # 再帰
    # MATLAB の n=2..maxdeg+2, k=n+1 に相当 → 0-based では k=n
    for n in range(2, maxdeg + 3):
        k = n  # 0-based index
        for m in range(0, n + 1):
            # MATLAB: p = m+1（第二添字は m に対応）→ 0-based では j=m
            j = m
            if n == m:
                # 対角
                # P[k,k] = sqrt(2n+1)/sqrt(2n) * sphi * P[k-1,k-1]
                P[k, k, :] = np.sqrt(2.0 * n + 1.0) / np.sqrt(2.0 * n) * sphi * P[k - 1, k - 1, :]
            elif m == 0:
                # P[k,0] = (sqrt(2n+1)/n) * ( sqrt(2n-1)*cphi*P[k-1,0] - (n-1)/sqrt(2n-3)*P[k-2,0] )
                P[k, 0, :] = (np.sqrt(2.0 * n + 1.0) / n) * (
                    np.sqrt(2.0 * n - 1.0) * cphi * P[k - 1, 0, :]
                    - (n - 1.0) / np.sqrt(2.0 * n - 3.0) * P[k - 2, 0, :]
                )
            else:
                # 一般項
                # P[k,j] = sqrt(2n+1)/(sqrt(n+m)*sqrt(n-m)) * ( sqrt(2n-1)*cphi*P[k-1,j]
                #            - sqrt(n+m-1)*sqrt(n-m-1)/sqrt(2n-3)*P[k-2,j] )
                P[k, j, :] = (
                    np.sqrt(2.0 * n + 1.0) / (np.sqrt(n + m) * np.sqrt(n - m))
                ) * (
                    np.sqrt(2.0 * n - 1.0) * cphi * P[k - 1, j, :]
                    - (np.sqrt(n + m - 1.0) * np.sqrt(n - m - 1.0) / np.sqrt(2.0 * n - 3.0))
                    * P[k - 2, j, :]
                )
    return P


def _compute_gravity(
    p: np.ndarray,
    maxdeg: int,
    P: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    smlambda: np.ndarray,
    cmlambda: np.ndarray,
    GM: float,
    Re: float,
    r: np.ndarray,
    scale_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB computeGravity と同等の数式を 0-based に移植。
    """
    # 安全な除算のため
    eps = 1e-16
    xy2 = p[:, 0] ** 2 + p[:, 1] ** 2
    rho = np.sqrt(xy2)
    rho_safe = np.where(rho < eps, eps, rho)

    rRatio = Re / r
    rRatio_n = rRatio.copy()

    dUdrSumN = np.ones_like(r)
    dUdphiSumN = np.zeros_like(r)
    dUdlambdaSumN = np.zeros_like(r)

    # MATLAB: for n = 2:maxdeg
    # 0-basedの P/C/S は [n+1, m] に相当する要素を参照するので
    # python の k = n でアクセスし、係数の (k+1) を使う
    for n in range(2, maxdeg + 1):
        k = n  # 0-based
        rRatio_n *= rRatio

        dUdrSumM = np.zeros_like(r)
        dUdphiSumM = np.zeros_like(r)
        dUdlambdaSumM = np.zeros_like(r)

        for m in range(0, n + 1):
            j = m  # 第二添字
            cos_sin = C[k, j] * cmlambda[:, j] + S[k, j] * smlambda[:, j]

            # dU/dr の和
            dUdrSumM += P[k, j, :] * cos_sin

            # dU/dphi の和
            # MATLAB: ( P(k,j+1,:)*scaleFactor(k,j,:) - p(:,3)/(sqrt(x^2+y^2))*m*P(k,j,:) ) * cos_sin
            # ここでは scale_factor[k, j] を使用。j+1 参照に注意（P shape は十分に確保している）
            pj1 = P[k, j + 1, :] if (j + 1) < P.shape[1] else 0.0
            dUdphiSumM += (pj1 * scale_factor[k, j] - (p[:, 2] / rho_safe) * m * P[k, j, :]) * cos_sin

            # dU/dlambda の和
            dUdlambdaSumM += m * P[k, j, :] * (S[k, j] * cmlambda[:, j] - C[k, j] * smlambda[:, j])

        dUdrSumN += dUdrSumM * rRatio_n * (k + 1)  # (k+1) は次数 n の「n+1」に対応
        dUdphiSumN += dUdphiSumM * rRatio_n
        dUdlambdaSumN += dUdlambdaSumM * rRatio_n

    # 球座標での勾配 → 物理加速度
    dUdr = -GM / (r ** 2) * dUdrSumN
    dUdphi = GM / r * dUdphiSumN
    dUdlambda = GM / r * dUdlambdaSumN

    # ECEF へ変換（Vallado 式）
    # 注意: rho→0 近傍のために rho_safe を使う
    ax = ((1.0 / r) * dUdr - (p[:, 2] / (r ** 2 * rho_safe)) * dUdphi) * p[:, 0] \
         - (dUdlambda / xy2.clip(min=eps)) * p[:, 1]
    ay = ((1.0 / r) * dUdr - (p[:, 2] / (r ** 2 * rho_safe)) * dUdphi) * p[:, 1] \
         + (dUdlambda / xy2.clip(min=eps)) * p[:, 0]
    az = (1.0 / r) * dUdr * p[:, 2] + (rho_safe / (r ** 2)) * dUdphi

    # 極（rho ~ 0）での特別扱い（MATLAB 実装と整合）
    at_pole = rho < 1e-12
    if np.any(at_pole):
        ax[at_pole] = 0.0
        ay[at_pole] = 0.0
        az[at_pole] = (1.0 / r[at_pole]) * dUdr[at_pole] * p[at_pole, 2]

    return ax, ay, az


# ==============================
# 2) EGM2008 ローダ（.mat）
# ==============================

@dataclass
class GravitySH:
    GM: float           # [m^3/s^2]
    Re: float           # [m]
    maxdeg: int
    C: np.ndarray       # shape: (maxdeg+3, maxdeg+3)  ※+2拡張
    S: np.ndarray       # shape: (maxdeg+3, maxdeg+3)
    sF: np.ndarray      # shape: (maxdeg+3, maxdeg+3)  （dUdphi 正規化用スケール）


def _legendre_scale_factor(maxdegree: int) -> np.ndarray:
    """
    MATLAB initgravitysphericalharmonic の loc_gravLegendre_scaleFactor と同等。
    dUdphi の正規化に使うスケール係数。
    返り値 shape = (maxdegree+3, maxdegree+3) で、_compute_legendre_polynomials と同一の添字系。
    """
    sf = np.zeros((maxdegree + 3, maxdegree + 3), dtype=float)

    # seeds（MATLAB の 1-based を 0-based に直したもの）
    # scaleFactor(1,1)=0; scaleFactor(2,1)=1; scaleFactor(2,2)=0
    sf[0, 0] = 0.0
    sf[1, 0] = 1.0
    sf[1, 1] = 0.0

    # MATLAB: for n=2:maxdeg+2, k=n+1 → 0-based: k=n
    for n in range(2, maxdegree + 3):
        k = n
        for m in range(0, n + 1):
            j = m
            if n == m:
                sf[k, k] = 0.0
            elif m == 0:
                # sqrt((n+1)*n/2)
                sf[k, j] = np.sqrt((n + 1) * n / 2.0)
            else:
                # sqrt((n+m+1)*(n-m))
                sf[k, j] = np.sqrt((n + m + 1) * (n - m))
    return sf


def init_gravity_spherical_harmonic(
    maxdeg: int,
    egm2008_mat: str = "EGM2008.mat",
    *,
    matlab_keys: Tuple[str, str, str, str, str] = ("GM", "Re", "degree", "C", "S"),
) -> GravitySH:
    """
    EGM2008 の .mat から係数を読み込み、最大次数/階 maxdeg まで切り出して返す。
    MATLAB initgravitysphericalharmonic.m と同等の挙動。

    Parameters
    ----------
    maxdeg : int
        使いたい次数・階（<= モデルの最大次数）
    egm2008_mat : str
        'GM', 'Re', 'degree', 'C', 'S' を含む MATLAB .mat のパス
    matlab_keys : tuple
        .mat 内のキー名（カスタム .mat の場合に上書き可能）

    Returns
    -------
    GravitySH
        GM, Re, maxdeg, C, S, sF を含むデータクラス
    """
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("scipy が必要です。requirements.txt に `scipy` を追加してください。") from e

    mat = loadmat(egm2008_mat)
    kGM, kRe, kDeg, kC, kS = matlab_keys

    for k in (kGM, kRe, kDeg, kC, kS):
        if k not in mat:
            raise KeyError(f"{egm2008_mat} に必要キー {matlab_keys} のいずれかが見つかりません。")

    GM = float(np.asarray(mat[kGM]).squeeze())
    Re = float(np.asarray(mat[kRe]).squeeze())
    degree_all = int(np.asarray(mat[kDeg]).squeeze())

    # MATLAB側は 1:maxdeg+2 の切り出しをしている（+2 拡張）
    use_deg = int(min(maxdeg, degree_all))
    ncut = use_deg + 2

    C_full = np.array(mat[kC], dtype=float)
    S_full = np.array(mat[kS], dtype=float)
    if C_full.shape[0] < ncut + 1 or C_full.shape[1] < ncut + 1:
        raise ValueError("EGM2008 C 行列のサイズが不足しています。")
    if S_full.shape[0] < ncut + 1 or S_full.shape[1] < ncut + 1:
        raise ValueError("EGM2008 S 行列のサイズが不足しています。")

    C = C_full[: ncut + 1, : ncut + 1].copy()
    S = S_full[: ncut + 1, : ncut + 1].copy()

    sF = _legendre_scale_factor(use_deg)

    return GravitySH(GM=GM, Re=Re, maxdeg=use_deg, C=C, S=S, sF=sF)


# 便利ラッパ：モデルを受け取って加速度を返す
def accel_from_model(rr_ecef: np.ndarray, model: GravitySH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_earth_gravitational_acceleration(
        rr_ecef,
        model.GM,
        model.Re,
        model.C,
        model.S,
        model.maxdeg,
        model.sF,
    )
