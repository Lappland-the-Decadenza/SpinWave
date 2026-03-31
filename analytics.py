import numpy as np
from numba import njit, prange
import contourpy
import core
import vertices

# =====================================================================
# КОНСТАНТЫ
# =====================================================================
ALPHA_G = 1e-4
L_PROP_M = 1e-6
NEPER_TO_DB = 20.0 / np.log(10.0)


# =====================================================================
# ВЫЧИСЛИТЕЛЬНОЕ ЯДРО NUMBA (С многопоточностью)
# =====================================================================

@njit(cache=True)
def _calculate_group_velocity_and_damping(k_mag, theta_k, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                                          omega_M_sqrt2, alpha_g):
    r"""Рассчитывает частоту, затухание и групповую скорость (v_g = d\omega / dk)."""
    Ak, _, omega = core.compute_light_mode_scalar(
        k_mag, theta_k, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2
    )
    gamma_k = alpha_g * Ak

    dk = max(k_mag * 1e-4, 1e-5)
    k_plus = k_mag + dk
    k_minus = max(k_mag - dk, 1e-9)

    _, _, omega_plus = core.compute_light_mode_scalar(k_plus, theta_k, d, alpha_ex, theta_M, omega_M, omega_H,
                                                      omega_M_half, omega_M_sqrt2)
    _, _, omega_minus = core.compute_light_mode_scalar(k_minus, theta_k, d, alpha_ex, theta_M, omega_M, omega_H,
                                                       omega_M_half, omega_M_sqrt2)

    actual_delta = k_plus - k_minus
    v_g_si = (omega_plus - omega_minus) / actual_delta * 0.01

    return omega, v_g_si, gamma_k


@njit(cache=True, parallel=True)
def _compute_gammas_for_arrays(k3_mag, k3_th, k4_mag, k4_th, d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half,
                               omega_M_sqrt2, alpha_g):
    """Многопоточный расчет затухания для массивов рассеянных магнонов."""
    n = len(k3_mag)
    g3 = np.empty(n, dtype=np.float64)
    g4 = np.empty(n, dtype=np.float64)

    for i in prange(n):
        Ak3, _, _ = core.compute_light_mode_scalar(k3_mag[i], k3_th[i], d, alpha_ex, theta_M, omega_M, omega_H,
                                                   omega_M_half, omega_M_sqrt2)
        Ak4, _, _ = core.compute_light_mode_scalar(k4_mag[i], k4_th[i], d, alpha_ex, theta_M, omega_M, omega_H,
                                                   omega_M_half, omega_M_sqrt2)
        g3[i] = alpha_g * Ak3
        g4[i] = alpha_g * Ak4

    return g3, g4


@njit(cache=True, parallel=True)
def _compute_thresholds_numba(gamma_3, gamma_4, W_abs):
    """Многопоточное вычисление порогов без выделения временной памяти NumPy."""
    n = len(W_abs)
    a_th2 = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if W_abs[i] > 1e-15:
            a_th2[i] = np.sqrt(gamma_3[i] * gamma_4[i]) / W_abs[i]
        else:
            a_th2[i] = np.inf

    return a_th2


# =====================================================================
# ОРКЕСТРАТОРЫ ДЛЯ ПАРАМЕТРОВ НАКАЧКИ И МОЩНОСТИ
# =====================================================================

def compute_pump_parameters(k_in_complex, state: core.SystemState):
    """Возвращает макроскопические параметры волны накачки."""
    k_in_mag = np.abs(k_in_complex)
    theta_k = np.angle(k_in_complex)

    omega, v_g_si, gamma_in = _calculate_group_velocity_and_damping(k_in_mag, theta_k, *state.numba_args, ALPHA_G)

    abs_v_g = np.abs(v_g_si)
    loss_db = NEPER_TO_DB * (gamma_in * L_PROP_M) / abs_v_g if abs_v_g > 1e-9 else np.nan

    return omega, v_g_si, gamma_in, loss_db


def calculate_threshold_power(a_th2, v_g_si, omega_in_rad_s, state: core.SystemState):
    """Вычисляет критическую мощность накачки (P_th) в Вт/м."""
    if np.isnan(a_th2):
        return np.nan

    v_g_cgs = np.abs(v_g_si) * 100.0
    P_th_cgs = (state.Ms * state.d * v_g_cgs * omega_in_rad_s * a_th2) / np.abs(core.GAMMA_CGS)

    return P_th_cgs * 1e-7 * 100.0


# =====================================================================
# АТОМАРНЫЕ ФУНКЦИИ АНАЛИЗА КОНТУРА
# =====================================================================

def _extract_zero_contour_vertices(K_x, K_y, E_mismatch):
    """
    Прямой вызов C++ движка contourpy для мгновенного извлечения геометрии.
    Обходит медленное создание визуальных графиков Matplotlib.
    """
    # Создаем математический генератор контуров
    generator = contourpy.contour_generator(K_x, K_y, E_mismatch)

    # Извлекаем координаты изолинии для уровня энергии 0.0
    # Возвращает список 2D массивов вершин
    lines = generator.lines(0.0)

    if not lines:
        return np.array([])

    # Исключаем вырожденные точки-сироты
    valid_paths = [p for p in lines if len(p) > 1]

    if not valid_paths:
        return np.array([])

    # Склеиваем все куски контура в один массив
    return np.concatenate(valid_paths)


def _handle_trivial_threshold(k_in_complex, gamma_in, state: core.SystemState):
    """Обрабатывает случай, когда контура рассеяния не существует."""
    k_arr = np.array([k_in_complex])
    W_trivial = np.abs(vertices.calculate_W_tilde((k_arr, k_arr, k_arr, k_arr), state)[0])

    a_th2 = gamma_in / W_trivial if W_trivial > 1e-15 else np.nan
    return a_th2, k_in_complex, k_in_complex


def _find_best_scattered_vectors(a_th2_array, k3_array, k4_array):
    """Находит векторы k3 и k4, соответствующие минимальному порогу."""
    min_a_th2 = np.nanmin(a_th2_array)

    if np.isnan(min_a_th2) or np.isinf(min_a_th2):
        return np.nan, k3_array[0], k4_array[0]

    tolerance = 1e-8 * min_a_th2 if min_a_th2 > 0 else 1e-12
    candidates = np.where(a_th2_array <= min_a_th2 + tolerance)[0]

    best_idx = candidates[np.argmin(np.real(k3_array[candidates]))]
    return a_th2_array[best_idx], k3_array[best_idx], k4_array[best_idx]


def find_minimum_threshold_on_contour(K_x, K_y, E_mismatch, k_in_complex, is_trivial, state: core.SystemState):
    """Главный оркестратор поиска порога."""
    _, _, gamma_in, _ = compute_pump_parameters(k_in_complex, state)

    if is_trivial:
        return _handle_trivial_threshold(k_in_complex, gamma_in, state)

    contour_vertices = _extract_zero_contour_vertices(K_x, K_y, E_mismatch)
    if len(contour_vertices) == 0:
        return np.nan, k_in_complex, k_in_complex

    k3_array = contour_vertices[:, 0] + 1j * contour_vertices[:, 1]
    k4_array = (2.0 * k_in_complex) - k3_array

    k1_array = np.full_like(k3_array, k_in_complex)
    k2_array = np.full_like(k3_array, k_in_complex)

    W_vals = vertices.calculate_W_tilde((k1_array, k2_array, k3_array, k4_array), state)
    W_abs = np.abs(W_vals)

    gamma_3, gamma_4 = _compute_gammas_for_arrays(
        np.abs(k3_array), np.angle(k3_array), np.abs(k4_array), np.angle(k4_array),
        *state.numba_args, ALPHA_G
    )

    a_th2_array = _compute_thresholds_numba(gamma_3, gamma_4, W_abs)

    return _find_best_scattered_vectors(a_th2_array, k3_array, k4_array)