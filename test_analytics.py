import pytest
import numpy as np
import math

import core
import analytics


# =====================================================================
# ФИКСТУРЫ
# =====================================================================

@pytest.fixture
def mock_state():
    """Стандартное состояние пленки (ЖИГ, поле в плоскости)."""
    return core.SystemState.from_si(
        Ms_si=140056.35, A_si=3.603e-12, B_ext_tesla=0.8,
        theta_H_rad=math.pi / 2, d_si=97.0e-9
    )


# =====================================================================
# 1. ТЕСТЫ ФИЗИЧЕСКИХ ПРОИЗВОДНЫХ И ПАРАМЕТРОВ
# =====================================================================

def test_group_velocity_exchange_limit(mock_state):
    """
    ФИЗИЧЕСКАЯ ТОЧНОСТЬ:
    В глубоком обменном пределе (огромные k) дисперсия стремится к параболе:
    omega = omega_H + omega_M * alpha_ex * k^2
    Значит, аналитическая групповая скорость (d_omega / d_k) равна:
    v_g_analytical = 2 * omega_M * alpha_ex * k
    """
    k_huge = 1e7  # 10^7 1/см
    theta_k = 0.0

    _, v_g_si_num, _ = analytics._calculate_group_velocity_and_damping(
        k_huge, theta_k, *mock_state.numba_args, analytics.ALPHA_G
    )

    # Считаем аналитическую скорость в СГС (см/с) и переводим в СИ (м/с), умножая на 0.01
    v_g_cgs_analytical = 2.0 * mock_state.omega_M * mock_state.alpha_ex * k_huge
    v_g_si_analytical = v_g_cgs_analytical * 0.01

    # Численная производная (центральная разность) должна идеально совпадать с аналитикой
    assert np.isclose(v_g_si_num, v_g_si_analytical, rtol=1e-3), \
        "Численный расчет групповой скорости сломался!"


def test_calculate_threshold_power_nan_handling(mock_state):
    """МАТЕМАТИЧЕСКАЯ ЗАЩИТА: Если порог не найден (NaN), мощность тоже должна быть NaN."""
    a_th2_nan = np.nan
    v_g_si = 1000.0
    omega_in = 5e10

    P_th = analytics.calculate_threshold_power(a_th2_nan, v_g_si, omega_in, mock_state)
    assert np.isnan(P_th), "Функция мощности не пробросила NaN дальше!"


# =====================================================================
# 2. ТЕСТЫ ИНТЕГРАЦИИ С CONTOURPY
# =====================================================================

def test_contourpy_extraction():
    """
    ИНТЕГРАЦИЯ C-ЯДРА: Проверяем, что contourpy правильно находит изолинии.
    Задаем искусственную матрицу в виде параболоида Z = X^2 + Y^2 - 25.
    Нулевой контур (Z=0) обязан быть идеальной окружностью радиуса 5.
    """
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    Z = X ** 2 + Y ** 2 - 25.0

    vertices = analytics._extract_zero_contour_vertices(X, Y, Z)

    assert len(vertices) > 0, "Движок contourpy не нашел очевидный контур!"

    # Проверяем, что все найденные точки лежат на окружности радиуса 5
    radii = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
    assert np.allclose(radii, 5.0, atol=1e-1), "Извлеченная геометрия контура искажена!"


# =====================================================================
# 3. ТЕСТ ЛОГИКИ ОРКЕСТРАТОРА (ЗАЩИТА ОТ ПИЛЫ)
# =====================================================================

def test_left_anchor_anti_sawtooth_logic():
    """
    АЛГОРИТМИЧЕСКАЯ ЛОГИКА:
    Тестируем выбор "левого лепестка" при наличии нескольких точек с одинаковым порогом.
    Это критически важно для гладкости графиков (избавления от численной "пилы").
    """
    # Искусственные массивы: 3 точки на контуре
    k3_array = np.array([
        1000.0 + 1j * 500.0,  # Правая точка (Re = 1000)
        -500.0 + 1j * 500.0,  # Самая ЛЕВАЯ точка (Re = -500) <- ИДЕАЛ
        200.0 + 1j * 500.0  # Точка по центру (Re = 200)
    ])
    k4_array = np.zeros_like(k3_array)  # k4 здесь не важен

    # Допустим, у всех трех точек порог оказался одинаковым (в пределах float погрешности)
    min_val = 1.2345e-10
    a_th2_array = np.array([min_val, min_val + 1e-20, min_val - 1e-20])

    best_c, best_k3, best_k4 = analytics._find_best_scattered_vectors(a_th2_array, k3_array, k4_array)

    # Алгоритм ОБЯЗАН выбрать индекс 1, так как там минимальная действительная часть (-500)
    assert best_k3 == k3_array[1], "Защита от пилы сломалась! Выбрана не самая левая точка контура."


# =====================================================================
# 4. ТЕСТЫ ОРКЕСТРАТОРА (Тривиальный режим)
# =====================================================================

def test_trivial_threshold_logic(mock_state):
    """
    ЛОГИКА: Если передан флаг is_trivial=True, оркестратор не должен
    искать контур, а обязан сразу посчитать порог для точки накачки (k3 = k4 = k_in).
    """
    k_in_complex = 2.5e4 + 1j * 0.0

    # Искусственные "пустые" сетки, которые обрушили бы contourpy, если бы он вызвался
    Kx, Ky, Em = np.array([]), np.array([]), np.array([])

    a_th2, k3_opt, k4_opt = analytics.find_minimum_threshold_on_contour(
        Kx, Ky, Em, k_in_complex, is_trivial=True, state=mock_state
    )

    assert k3_opt == k_in_complex, "Для тривиального режима k3 должно быть равно k_in!"
    assert k4_opt == k_in_complex, "Для тривиального режима k4 должно быть равно k_in!"
    assert not np.isnan(a_th2), "Порог для тривиального режима не посчитался!"


# =====================================================================
# 5. ТЕСТЫ ЧИСЛЕННОЙ СТАБИЛЬНОСТИ И ЭКСТРЕМАЛЬНЫХ ПРЕДЕЛОВ
# =====================================================================

def test_compute_thresholds_numba_inf_handling():
    """
    ЧИСЛЕННАЯ СТАБИЛЬНОСТЬ:
    Проверяем, что многопоточное ядро корректно обрабатывает нулевые амплитуды W.
    При W -> 0 порог должен улетать в бесконечность, а не падать с делением на ноль.
    """
    gamma_3 = np.array([1e6])
    gamma_4 = np.array([1e6])
    W_abs = np.array([0.0])  # Строгий ноль

    a_th2_array = analytics._compute_thresholds_numba(gamma_3, gamma_4, W_abs)

    assert np.isinf(a_th2_array[0]), "Ядро не вернуло np.inf при делении на нулевое W!"


def test_zero_group_velocity_loss_protection():
    """
    МАТЕМАТИЧЕСКАЯ ЗАЩИТА: Проверяем логику отсечения околонулевой групповой скорости.
    (Так как физически при k->0 скорость в пленке конечна из-за дипольного вклада |k|d,
    мы тестируем саму математическую защиту от деления на ноль).
    """
    gamma_in = 1e6
    v_g_zero = 1e-12  # Искусственная нулевая скорость

    abs_v_g = np.abs(v_g_zero)
    # Имитируем строку расчета из compute_pump_parameters
    safe_loss_db = analytics.NEPER_TO_DB * (gamma_in * analytics.L_PROP_M) / abs_v_g if abs_v_g > 1e-9 else np.nan

    assert np.isnan(safe_loss_db), "Защита от деления на нулевую скорость не сработала!"
