import math
import numpy as np
from scipy.optimize import newton, root_scalar
from numba import njit
import core


@njit(cache=True)
def safe_divide(n, d):
    """
    Универсальное безопасное деление.
    Работает и в njit-ядрах, и в обычном коде.
    """
    if np.abs(d) > 1e-9:
        return n / d
    return 0.0j


def generate_center_dense_grid(center, min_val, max_val, res, power=3.0):
    """
    Создает 1D сетку с экспоненциальным сгущением точек вокруг центра.
    Полезно для детального разрешения областей вблизи k_in.
    """
    t = np.linspace(-1, 1, res)
    t_dense = np.sign(t) * (np.abs(t) ** power)
    arr = np.where(t_dense < 0,
                   center + t_dense * (center - min_val),
                   center + t_dense * (max_val - center))
    return arr


# =====================================================================
# РЕШАТЕЛИ (Поиск корней уравнений)
# =====================================================================

def find_k_for_ghz(f_ghz: float, theta_k_rad: float, state: core.SystemState, k_guess: float = 1e4) -> float:
    """
    Ищет волновой вектор k (в СГС) для заданной частоты.
    Возвращает скалярное значение k.
    """
    target_omega = 2.0 * math.pi * f_ghz * 1e9
    args = state.numba_args

    def objective(k_val):
        # Используем быструю скалярную функцию
        _, _, om = core.compute_light_mode_scalar(k_val, theta_k_rad, *args)
        return om - target_omega

    try:
        return float(newton(objective, x0=k_guess))
    except RuntimeError:
        return np.nan


def find_He_for_ghz(f_ghz: float, k_si: float, theta_k_rad: float, Ms_si: float,
                    A_si: float, theta_H_rad: float, d_si: float, He_guess: float = None) -> float:
    """
    Ищет внешнее магнитное поле He (в Теслах) для удержания заданного k на нужной частоте.
    Возвращает скалярное значение поля He.
    """
    target_omega = 2.0 * math.pi * f_ghz * 1e9
    k_cgs = k_si * core.SI_TO_CGS_K
    guess = f_ghz / 28.0 if He_guess is None else He_guess

    def objective(He_tesla):
        He_eff = max(He_tesla, 1e-6)
        temp_state = core.SystemState.from_si(Ms_si, A_si, He_eff, theta_H_rad, d_si)
        _, _, om = core.compute_light_mode_scalar(k_cgs, theta_k_rad, *temp_state.numba_args)
        return om - target_omega

    try:
        # Метод Ньютона (быстрый)
        return abs(newton(objective, x0=guess, tol=1e-4, maxiter=30))
    except (RuntimeError, OverflowError):
        pass

    try:
        # Запасной метод Брента (медленный, но железобетонный)
        if objective(1e-4) > 0:
            return 0.0
        res = root_scalar(objective, bracket=[1e-4, 10.0], method='brentq')
        return res.root
    except ValueError:
        return np.nan