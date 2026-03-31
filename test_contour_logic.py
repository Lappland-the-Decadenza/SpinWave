import pytest
import numpy as np
import math

import core
import contour_logic
import utils

# =====================================================================
# ФИКСТУРЫ И БАЗОВЫЕ СОСТОЯНИЯ
# =====================================================================

PARAMS_SI = {
    'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'He': 0.8
}


@pytest.fixture
def state_in_plane():
    """Магнитное поле в плоскости (90 градусов) -> Контур обычно ЕСТЬ."""
    return core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], PARAMS_SI['He'],
        theta_H_rad=math.pi / 2, d_si=PARAMS_SI['d']
    )


@pytest.fixture
def state_out_of_plane():
    """Магнитное поле перпендикулярно (0 градусов) -> Контура НЕТ (тривиальный режим)."""
    return core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], PARAMS_SI['He'],
        theta_H_rad=0.0, d_si=PARAMS_SI['d']
    )


# =====================================================================
# 1. ТЕСТЫ ГРАНИЦ И РЕЖИМОВ (Тривиальный / Не тривиальный)
# =====================================================================

def test_trivial_regime_fallback(state_out_of_plane):
    """
    Проверка тривиального режима (когда контура физически нет).
    Чтобы гарантированно попасть в тривиальный режим, уйдем в глубокий обменный предел,
    где дисперсия является строгой выпуклой параболой, и 4-магнонное рассеяние кинематически запрещено.
    """
    # 1. Тест с огромным вектором (чистый обменный предел)
    k_in_mag = 1e7
    k_in_complex = k_in_mag * np.exp(1j * 0.0)

    k_span, is_trivial = contour_logic.find_contour_boundaries(k_in_complex, state_out_of_plane)

    assert is_trivial is True, "В глубоком обменном пределе контура быть не должно!"
    assert np.isclose(k_span, k_in_mag * 1.5), "Размер окна должен быть равен 1.5 * k_in"

    # 2. Тест с искусственным состоянием без дипольного вклада (ультра-тонкая пленка d -> 0)
    # Здесь дисперсия выпуклая даже при малых k
    pure_exchange_state = core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], PARAMS_SI['He'], 0.0, 1e-12
    )
    k_tiny = 1e-2
    k_span_tiny, is_trivial_tiny = contour_logic.find_contour_boundaries(k_tiny + 0j, pure_exchange_state)

    assert is_trivial_tiny is True, "В тонкой пленке без дипольного вклада контура быть не должно!"
    assert np.isclose(k_span_tiny, 1e4), "Защита 1e4 (размер окна по умолчанию) не сработала для малых k!"


def test_nontrivial_contour_found(state_in_plane):
    """
    Проверка нетривиального режима (когда контур физически существует).
    Для In-Plane геометрии и k_in = 5e4 cgs контур точно должен разлететься.
    """
    k_in_complex = 5e4 * np.exp(1j * 0.0)

    k_span, is_trivial = contour_logic.find_contour_boundaries(k_in_complex, state_in_plane)

    assert is_trivial is False, "Алгоритм не нашел существующий контур!"
    assert k_span > np.abs(k_in_complex), "Размах контура должен быть больше самого вектора накачки"


# =====================================================================
# 2. ТЕСТЫ СИНГУЛЯРНОСТЕЙ И ТОЧНОСТИ (Numba-ядра)
# =====================================================================

def test_forward_scattering_energy_conservation(state_in_plane):
    """
    Математическая точность: В точке тривиального рассеяния (k3 = k_in, k4 = k_in)
    невязка энергии должна быть ИДЕАЛЬНЫМ НУЛЕМ.
    """
    k_in_complex = 5e4 + 1j * 1e4
    kx_in, ky_in = np.real(k_in_complex), np.imag(k_in_complex)

    _, _, om_in = core.compute_light_mode_scalar(np.abs(k_in_complex), np.angle(k_in_complex),
                                                 *state_in_plane.numba_args)
    E_target = 2.0 * om_in

    # Считаем невязку точно в точке k_in
    mismatch = contour_logic._calculate_mismatch_point(
        kx_in, ky_in, kx_in, ky_in, E_target, *state_in_plane.numba_args
    )

    assert np.isclose(mismatch, 0.0,
                      atol=1e-12), f"Энергия не сохраняется при тривиальном рассеянии! Ошибка: {mismatch}"


def test_origin_singularity_protection(state_in_plane):
    """
    Проверка того самого "магического" числа 5.0.
    Если k3 или k4 попадают в начало координат, функция обязана вернуть NaN.
    """
    k_in_complex = 1e4 + 1j * 0.0
    kx_in, ky_in = np.real(k_in_complex), np.imag(k_in_complex)
    E_target = 1e10  # Произвольная цель

    # 1. Точка k3 = 0 (попадание прямо в центр)
    val_k3_zero = contour_logic._calculate_mismatch_point(0.0, 0.0, kx_in, ky_in, E_target, *state_in_plane.numba_args)
    assert np.isnan(val_k3_zero), "Сингулярность k3=0 не была отловлена!"

    # 2. Точка k4 = 0 (это происходит, когда k3 = 2 * k_in)
    val_k4_zero = contour_logic._calculate_mismatch_point(2 * kx_in, 2 * ky_in, kx_in, ky_in, E_target,
                                                          *state_in_plane.numba_args)
    assert np.isnan(val_k4_zero), "Сингулярность k4=0 не была отловлена!"

    # 3. Нормальная точка
    val_normal = contour_logic._calculate_mismatch_point(kx_in, ky_in, kx_in, ky_in, E_target,
                                                         *state_in_plane.numba_args)
    assert not np.isnan(val_normal), "Нормальная точка ошибочно получила NaN!"


# =====================================================================
# 3. ТЕСТЫ ГЕНЕРАЦИИ СЕТКИ И УТИЛИТ
# =====================================================================

def test_compute_mismatch_grid_shapes(state_in_plane):
    """
    Интеграционный тест этапа 2: проверяем форму возвращаемых массивов.
    """
    k_in_complex = 5e4 + 1j * 0.0
    k_span = 1e5
    grid_res = 50

    K_x, K_y, E_miss = contour_logic.compute_mismatch_grid(k_in_complex, k_span, state_in_plane, grid_res)

    expected_shape = (grid_res, grid_res)
    assert K_x.shape == expected_shape, f"Неверная форма K_x: {K_x.shape}"
    assert K_y.shape == expected_shape, f"Неверная форма K_y: {K_y.shape}"
    assert E_miss.shape == expected_shape, f"Неверная форма E_mismatch: {E_miss.shape}"


# =====================================================================
# 4. ПРОДВИНУТЫЕ ТЕСТЫ СЕТОК И ВНУТРЕННИХ ЯДЕР
# =====================================================================

def test_mismatch_grid_symmetry(state_in_plane):
    """
    ФИЗИЧЕСКАЯ СИММЕТРИЯ СЕТКИ:
    Для поля и накачки вдоль оси X (theta = 0), матрица невязки энергии
    должна быть идеально симметрична относительно горизонтальной оси (k_y = 0).
    """
    k_in_complex = 5e4 + 1j * 0.0  # Строго вдоль оси X
    k_span = 1e5
    grid_res = 31  # Нечетное число, чтобы центральная ось точно попала в индекс

    _, _, E_miss = contour_logic.compute_mismatch_grid(k_in_complex, k_span, state_in_plane, grid_res)

    # E_miss может содержать NaN в центре (защита от сингулярностей),
    # поэтому для сравнения заменяем NaN на нули
    E_miss_safe = np.nan_to_num(E_miss, nan=0.0)

    # Зеркально отражаем матрицу по вертикали (вверх ногами)
    E_miss_flipped = np.flipud(E_miss_safe)

    np.testing.assert_allclose(E_miss_safe, E_miss_flipped, atol=1e-8,
                               err_msg="Матрица невязки потеряла физическую симметрию по Y!")


def test_grid_center_exactness(state_in_plane):
    """
    СТРУКТУРНАЯ ТОЧНОСТЬ:
    При нечетном разрешении сетки центральный элемент матриц (K_x, K_y)
    обязан идеально совпадать с вектором накачки, а невязка в нем должна быть нулем.
    """
    k_in_complex = -3e4 + 1j * 7e4  # Произвольный асимметричный вектор
    grid_res = 51  # Нечетное -> центр имеет индекс 25
    center_idx = grid_res // 2

    K_x, K_y, E_miss = contour_logic.compute_mismatch_grid(k_in_complex, 1e5, state_in_plane, grid_res)

    kx_center = K_x[center_idx, center_idx]
    ky_center = K_y[center_idx, center_idx]
    e_center = E_miss[center_idx, center_idx]

    assert np.isclose(kx_center, np.real(k_in_complex)), "Центр K_x смещен!"
    assert np.isclose(ky_center, np.imag(k_in_complex)), "Центр K_y смещен!"
    assert np.isclose(e_center, 0.0, atol=1e-10), "Энергия в центре сетки не сохраняется!"


def test_ray_search_no_root(state_out_of_plane):
    """
    ВНУТРЕННЕЕ ПОКРЫТИЕ (Edge Case):
    Прямой вызов Numba-ядра _find_root_along_ray в условиях, когда корня точно нет.
    Функция должна безопасно вернуть -1.0, отработав свои 100 шагов.
    """
    # Загоняем в глубокий обменный предел
    k_in = 1e8 + 0j
    _, _, om_in = core.compute_light_mode_scalar(1e8, 0.0, *state_out_of_plane.numba_args)

    # ВАЖНО: передаем аргументы строго позиционно
    root = contour_logic._find_root_along_ray(
        k_in, math.pi, 100.0, 2.0 * om_in,
        *state_out_of_plane.numba_args
    )

    assert root == -1.0, f"Функция нашла фантомный корень там, где его нет: {root}"