import math
import numpy as np
import logging
from scipy.optimize import newton, minimize_scalar
from scipy import constants as const
from numba import njit

logger = logging.getLogger(__name__)

# =====================================================================
# ФИЗИЧЕСКИЕ КОНСТАНТЫ И КОЭФФИЦИЕНТЫ КОНВЕРТАЦИИ
# =====================================================================
SQRT_2 = np.sqrt(2.0)

SI_TO_CGS_A = 1e5
SI_TO_CGS_MS = 1e-3
SI_TO_CGS_FIELD = 1e4
SI_TO_CGS_D = 100.0
SI_TO_CGS_K = 0.01

HBAR_CGS = const.hbar * 1e7
MU_B_CGS = const.physical_constants['Bohr magneton'][0] * 1e3
GAMMA_CGS = (2 * MU_B_CGS) / HBAR_CGS


# =====================================================================
# 1. СОСТОЯНИЕ СИСТЕМЫ
# =====================================================================

class SystemState:
    """Хранит статические параметры намагниченной пленки."""
    __slots__ = [
        'Ms', 'd', 'alpha_ex', 'theta_M',
        'omega_M', 'omega_H', 'omega_M_half', 'omega_M_sqrt2'
    ]

    def __init__(self, Ms_cgs: float, A_cgs: float, He_cgs: float, theta_H_rad: float, d_cgs: float):
        self.Ms = Ms_cgs
        self.d = d_cgs
        self.alpha_ex = A_cgs / (2 * math.pi * self.Ms ** 2)

        self.theta_M = self._solve_theta_M(He_cgs, theta_H_rad)

        self.omega_M = abs(GAMMA_CGS) * 4 * math.pi * self.Ms
        self.omega_H = abs(GAMMA_CGS) * He_cgs * math.cos(self.theta_M - theta_H_rad) - self.omega_M * math.cos(self.theta_M) ** 2

        self.omega_M_half = self.omega_M / 2.0
        self.omega_M_sqrt2 = self.omega_M / SQRT_2

    @property
    def numba_args(self):
        """Удобная распаковка для передачи параметров состояния в Numba."""
        return (self.d, self.alpha_ex, self.theta_M, self.omega_M,
                self.omega_H, self.omega_M_half, self.omega_M_sqrt2)

    @classmethod
    def from_si(cls, Ms_si: float, A_si: float, B_ext_tesla: float, theta_H_rad: float, d_si: float):
        return cls(
            Ms_cgs=Ms_si * SI_TO_CGS_MS, A_cgs=A_si * SI_TO_CGS_A,
            He_cgs=B_ext_tesla * SI_TO_CGS_FIELD, theta_H_rad=theta_H_rad, d_cgs=d_si * SI_TO_CGS_D
        )

    def _solve_theta_M(self, He: float, theta_H: float) -> float:
        K_eq = 2.0 * math.pi * self.Ms
        K_der = 4.0 * math.pi * self.Ms
        K_der2 = 8.0 * math.pi * self.Ms

        grid = np.linspace(-math.pi / 2, math.pi / 2, 90)
        energies = -He * np.cos(grid - theta_H) - K_eq * np.sin(grid) ** 2
        best_guess = grid[np.argmin(energies)]

        def eq(theta):
            return He * math.sin(theta - theta_H) - K_eq * math.sin(2.0 * theta)

        def der(theta):
            return He * math.cos(theta - theta_H) - K_der * math.cos(2.0 * theta)

        def der2(theta):
            return -He * math.sin(theta - theta_H) + K_der2 * math.sin(2.0 * theta)

        try:
            theta_res = newton(eq, x0=best_guess, fprime=der, fprime2=der2, maxiter=10)
            if der(theta_res) > 0:
                return theta_res
        except RuntimeError:
            logger.debug("Ньютон не сошелся для theta_M, переключаемся на minimize_scalar.")

        def energy_scalar(theta):
            return -He * math.cos(theta - theta_H) - K_eq * math.sin(theta) ** 2

        res = minimize_scalar(energy_scalar, bounds=(-math.pi / 2, math.pi / 2), method='bounded')
        return res.x


# =====================================================================
# 2. ВЫЧИСЛИТЕЛЬНОЕ ЯДРО NUMBA (Только скалярные версии)
# =====================================================================

@njit(cache=True)
def compute_heavy_mode_scalar(ki, thi, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    """Вычисляет параметры для одного вектора k."""
    c_M = np.cos(theta_M)
    s_M = np.sin(theta_M)
    c_M_sq = c_M ** 2
    s_M_sq = s_M ** 2

    kd = ki * d
    if np.abs(kd) < 1e-8:
        gk = kd / 2.0
    else:
        gk = 1.0 - (1.0 - np.exp(-kd)) / kd

    c_k = np.cos(thi)
    s_k = np.sin(thi)

    k_term = alpha_ex * ki ** 2
    gk_ck2 = gk * c_k ** 2
    one_minus_gk = 1.0 - gk
    gk_sk_ck = gk * s_k * c_k

    Nxx = k_term + gk_ck2 * c_M_sq + one_minus_gk * s_M_sq
    Nyy = k_term + gk * s_k ** 2
    Nzz = k_term + gk_ck2 * s_M_sq + one_minus_gk * c_M_sq
    Nxy = gk_sk_ck * c_M
    Nxz = (gk_ck2 - one_minus_gk) * s_M * c_M
    Nyz = gk_sk_ck * s_M

    Ak = omega_H + omega_M_half * (Nxx + Nyy)
    Bk_i = omega_M_half * (-Nxx + Nyy + 2j * Nxy)
    Qk = omega_M_half * (Nxx + Nyy)
    Dk = omega_M_sqrt2 * (1j * Nxz + Nyz)
    Gamma_zz = omega_M * Nzz

    val_disp = Ak ** 2 - (Bk_i.real ** 2 + Bk_i.imag ** 2)

    if Ak > 0.0 and val_disp > 0.0:
        omega = np.sqrt(val_disp)
    else:
        omega = 0.0

    if omega > 1e-10:
        val_u = (Ak + omega) / (2 * omega)
        uk = np.sqrt(val_u) if val_u > 0 else 0.0

        abs_Bk = np.sqrt(Bk_i.real ** 2 + Bk_i.imag ** 2)
        if abs_Bk > 1e-10:
            val_v = (Ak - omega) / (2 * omega)
            vk_mag = np.sqrt(val_v) if val_v > 0 else 0.0
            vk = (-Bk_i / abs_Bk) * vk_mag
        else:
            vk = 0.0j
    else:
        uk = 0.0
        vk = 0.0j

    return uk, vk, Dk, Bk_i, Qk, Gamma_zz, omega


@njit(cache=True)
def compute_light_mode_scalar(ki, thi, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    """Легкая функция дисперсии для одного числа (возвращает Ak, Bk, omega)."""
    kd = ki * d
    if np.abs(kd) < 1e-8:
        gk = kd / 2.0
    else:
        gk = 1.0 - (1.0 - np.exp(-kd)) / kd

    c_k = np.cos(thi)
    s_k = np.sin(thi)
    c_M_sq = np.cos(theta_M) ** 2
    s_M_sq = np.sin(theta_M) ** 2
    c_M = np.cos(theta_M)

    k_term = alpha_ex * ki ** 2
    gk_ck2 = gk * c_k ** 2
    one_minus_gk = 1.0 - gk
    gk_sk_ck = gk * s_k * c_k

    Nxx = k_term + gk_ck2 * c_M_sq + one_minus_gk * s_M_sq
    Nyy = k_term + gk * s_k ** 2
    Nxy = gk_sk_ck * c_M

    Ak_i = omega_H + omega_M_half * (Nxx + Nyy)
    Bk_i = omega_M_half * (-Nxx + Nyy + 2j * Nxy)

    val_disp = Ak_i ** 2 - (Bk_i.real ** 2 + Bk_i.imag ** 2)

    if Ak_i > 0.0 and val_disp > 0.0:
        omega = np.sqrt(val_disp)
    else:
        omega = 0.0

    return Ak_i, Bk_i, omega