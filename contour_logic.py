import numpy as np
from numba import njit, prange
import core
import utils


# =====================================================================
# NUMBA-ЯДРА: ПОИСК ГРАНИЦ (Разбито на атомарные функции)
# =====================================================================

@njit(cache=True)
def _bisection_search(a, b, k_in_complex, ang, E_target, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                      omega_M_sqrt2):
    """Шаг 3: Уточняет корень методом бисекции на найденном отрезке [a, b]."""
    for _ in range(50):
        mid = (a + b) / 2.0
        k3_mid = k_in_complex + mid * np.exp(1j * ang)
        k4_mid = 2.0 * k_in_complex - k3_mid

        _, _, om3_mid = core.compute_light_mode_scalar(np.abs(k3_mid), np.angle(k3_mid), d, alpha_ex, theta_M, omega_M,
                                                       omega_H, omega_M_half, omega_M_sqrt2)
        _, _, om4_mid = core.compute_light_mode_scalar(np.abs(k4_mid), np.angle(k4_mid), d, alpha_ex, theta_M, omega_M,
                                                       omega_H, omega_M_half, omega_M_sqrt2)

        val_mid = (om3_mid + om4_mid) - E_target

        if val_mid > 0.0:
            b = mid
        else:
            a = mid

    return (a + b) / 2.0


@njit(cache=True)
def _find_root_along_ray(k_in_complex, ang, initial_r, E_target, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                         omega_M_sqrt2):
    """Шаг 2: Идет по конкретному лучу экспоненциальным шагом, находит вилку и запускает бисекцию."""
    r_val = initial_r
    found_negative_trench = False

    for step in range(100):
        k3_c = k_in_complex + r_val * np.exp(1j * ang)
        k4_c = 2.0 * k_in_complex - k3_c

        _, _, om3 = core.compute_light_mode_scalar(np.abs(k3_c), np.angle(k3_c), d, alpha_ex, theta_M, omega_M, omega_H,
                                                   omega_M_half, omega_M_sqrt2)
        _, _, om4 = core.compute_light_mode_scalar(np.abs(k4_c), np.angle(k4_c), d, alpha_ex, theta_M, omega_M, omega_H,
                                                   omega_M_half, omega_M_sqrt2)

        val = (om3 + om4) - E_target

        # Ищем переход от отрицательной невязки к положительной (пересечение нуля)
        if val < 0.0:
            found_negative_trench = True

        if found_negative_trench and val > 0.0:
            # Вилка найдена (между r_val/2 и r_val)! Запускаем точный поиск.
            return _bisection_search(r_val / 2.0, r_val, k_in_complex, ang, E_target,
                                     d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)

        r_val *= 2.0

    return -1.0  # Корень на этом луче не найден


@njit(cache=True)
def _find_boundaries_numba_kernel(k_in_complex, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    """Шаг 1: Главное ядро поиска границ контура. Сканирует пространство радиальными лучами."""
    k_in_mag = np.abs(k_in_complex)
    ang_in = np.angle(k_in_complex)

    _, _, om_in = core.compute_light_mode_scalar(k_in_mag, ang_in, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                                                 omega_M_sqrt2)
    E_target = 2.0 * om_in

    RAY_COUNT = 32
    max_root = -1.0
    initial_r = max(k_in_mag * 1e-4, 1.0)

    for i in range(RAY_COUNT):
        ang = 2.0 * np.pi * i / RAY_COUNT
        root = _find_root_along_ray(k_in_complex, ang, initial_r, E_target,
                                    d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        if root > max_root:
            max_root = root

    return max_root


# =====================================================================
# NUMBA-ЯДРА: ГЕНЕРАЦИЯ СЕТКИ
# =====================================================================

@njit(cache=True)
def _calculate_mismatch_point(k3x, k3y, kx_in, ky_in, E_target, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                              omega_M_sqrt2):
    """Вычисляет невязку энергии для одной конкретной точки на сетке."""
    k4x = 2.0 * kx_in - k3x
    k4y = 2.0 * ky_in - k3y

    k3_mag = np.sqrt(k3x ** 2 + k3y ** 2)
    k4_mag = np.sqrt(k4x ** 2 + k4y ** 2)

    # Исключаем околонулевые векторы (физическая сингулярность)
    if k3_mag < 5.0 or k4_mag < 5.0:
        return np.nan

    k3_th = np.arctan2(k3y, k3x)
    k4_th = np.arctan2(k4y, k4x)

    _, _, om3 = core.compute_light_mode_scalar(k3_mag, k3_th, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                                               omega_M_sqrt2)
    _, _, om4 = core.compute_light_mode_scalar(k4_mag, k4_th, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                                               omega_M_sqrt2)

    return (om3 + om4) - E_target


@njit(cache=True, parallel=True)  # <-- Включаем параллелизм
def _mismatch_grid_numba_kernel(Kx, Ky, kx_in, ky_in, E_target, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    """Многопоточный обход 2D сетки для вычисления энергии."""
    rows, cols = Kx.shape
    E_mismatch = np.empty((rows, cols), dtype=np.float64)

    # Используем prange только для внешнего цикла!
    for i in prange(rows):
        for j in range(cols):
            E_mismatch[i, j] = _calculate_mismatch_point(
                Kx[i, j], Ky[i, j], kx_in, ky_in, E_target,
                d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2
            )

    return E_mismatch

# =====================================================================
# ОБЕРТКИ PYTHON
# =====================================================================

def find_contour_boundaries(k_in_complex, state: core.SystemState):
    """ Этап 1: Молниеносный поиск границ контура """
    k_in_mag = np.abs(k_in_complex)
    max_radius_found = _find_boundaries_numba_kernel(k_in_complex, *state.numba_args)

    if max_radius_found > 0:
        k_span = max(max_radius_found, k_in_mag) * 1.2
        return k_span, False
    else:
        return max(k_in_mag * 1.5, 1e4), True


def compute_mismatch_grid(k_in_complex, k_span, state: core.SystemState, grid_res):
    """ Этап 2: Вычисление энергии на сетке """

    # Целевая энергия (через скалярную функцию)
    _, _, om_in = core.compute_light_mode_scalar(
        np.abs(k_in_complex), np.angle(k_in_complex), *state.numba_args
    )
    E_target = 2.0 * om_in

    kx_in, ky_in = np.real(k_in_complex), np.imag(k_in_complex)

    # Используем утилиту для генерации сетки со сгущением к центру
    k_vals_x = utils.generate_center_dense_grid(kx_in, kx_in - k_span, kx_in + k_span, grid_res)
    k_vals_y = utils.generate_center_dense_grid(ky_in, ky_in - k_span, ky_in + k_span, grid_res)
    K_x, K_y = np.meshgrid(k_vals_x, k_vals_y)

    E_mismatch = _mismatch_grid_numba_kernel(
        K_x, K_y, kx_in, ky_in, E_target, *state.numba_args
    )

    return K_x, K_y, E_mismatch
