import math
import numpy as np
import pytest

import core
import utils  # Подключаем наши решатели

# =====================================================================
# ФИКСТУРЫ И БАЗОВЫЕ СОСТОЯНИЯ
# =====================================================================

PARAMS_SI = {
    'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'He': 0.8
}


@pytest.fixture
def state_in_plane():
    """Магнитное поле в плоскости (90 градусов)."""
    return core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], PARAMS_SI['He'],
        theta_H_rad=math.pi / 2, d_si=PARAMS_SI['d']
    )


@pytest.fixture
def state_out_of_plane():
    """Магнитное поле перпендикулярно пленке (0 градусов)."""
    return core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], PARAMS_SI['He'],
        theta_H_rad=0.0, d_si=PARAMS_SI['d']
    )


# =====================================================================
# ТЕСТЫ 1: ФИЗИЧЕСКИЕ АСИМПТОТИКИ И ПРЕДЕЛЫ
# =====================================================================

def test_exchange_limit_asymptotics(state_in_plane):
    """
    ПРОВЕРКА ОБМЕННОГО ПРЕДЕЛА:
    При огромных k (k -> infty) диполь-дипольным взаимодействием можно пренебречь.
    Частота должна стремиться к аналитической формуле: omega = omega_H + omega_M * alpha_ex * k^2
    """
    k_huge = 1e8  # 10^8 1/см
    theta_k = 0.0

    _, _, om_num = core.compute_light_mode_scalar(k_huge, theta_k, *state_in_plane.numba_args)

    om_analytical = state_in_plane.omega_H + state_in_plane.omega_M * state_in_plane.alpha_ex * (k_huge ** 2)

    # Относительная погрешность должна быть крошечной (менее 0.1%)
    assert np.isclose(om_num, om_analytical, rtol=1e-3), \
        f"Обменный предел сломан! Численно: {om_num}, Аналитически: {om_analytical}"


def test_bogoliubov_identity(state_in_plane):
    """
    КВАНТОВОМЕХАНИЧЕСКАЯ НОРМИРОВКА:
    Для магнонов (бозонов) каноническое преобразование Боголюбова строго требует: u_k^2 - |v_k|^2 = 1.
    """
    k_mag = 5e5
    theta_k = math.pi / 3

    uk, vk, *_ = core.compute_heavy_mode_scalar(k_mag, theta_k, *state_in_plane.numba_args)

    bogoliubov_invariant = uk ** 2 - np.abs(vk) ** 2
    np.testing.assert_allclose(bogoliubov_invariant, 1.0, atol=1e-10,
                               err_msg="Нарушено условие каноничности: u^2 - |v|^2 != 1")


def test_kittel_fmr_limits(state_in_plane, state_out_of_plane):
    """При k -> 0 частота должна совпадать с классической формулой Киттеля для ФМР."""
    k_zero = 1e-8

    # 1. In-plane
    _, _, om_in = core.compute_light_mode_scalar(k_zero, 0.0, *state_in_plane.numba_args)
    w_H_in = state_in_plane.omega_H
    w_M_in = state_in_plane.omega_M
    kittel_in = np.sqrt(w_H_in * (w_H_in + w_M_in))
    np.testing.assert_allclose(om_in, kittel_in, rtol=1e-4, err_msg="In-plane ФМР не совпадает с Киттелем")

    # 2. Out-of-plane
    _, _, om_out = core.compute_light_mode_scalar(k_zero, 0.0, *state_out_of_plane.numba_args)
    w_H_out = state_out_of_plane.omega_H
    kittel_out = w_H_out
    np.testing.assert_allclose(om_out, kittel_out, rtol=1e-4, err_msg="Out-of-plane ФМР не совпадает с Киттелем")


# =====================================================================
# ТЕСТЫ 2: МАТЕМАТИЧЕСКИЕ ПОДВОХИ И ДЕЛЕНИЕ НА НОЛЬ
# =====================================================================

def test_ultra_thin_film_zero_division(state_in_plane):
    """
    ЗАЩИТА ОТ ДЕЛЕНИЯ НА НОЛЬ:
    Формула 1 - (1 - e^-kd)/kd при kd -> 0 дает неопределенность 0/0.
    Код должен использовать разложение Тейлора (gk = kd / 2) и не падать с NaN.
    """
    k_mag = 1e2
    d_nano = 1e-15  # Нереально тонкая пленка (почти 0)

    # Подменяем аргумент d в кортеже
    args = list(state_in_plane.numba_args)
    args[0] = d_nano

    Ak, Bk, om = core.compute_light_mode_scalar(k_mag, 0.0, *args)

    assert not np.isnan(om), "Частота ушла в NaN из-за деления на ноль!"
    assert not np.isnan(Ak), "Ak ушел в NaN!"


def test_negative_dispersion_handling():
    """
    ЗАЩИТА ОТ КОМПЛЕКСНЫХ ЧАСТОТ (НЕСТАБИЛЬНОСТЬ):
    Подаем заведомо отрицательное эффективное поле напрямую в ядро,
    минуя умный класс SystemState (который иначе перевернул бы намагниченность).
    """
    # Искусственно задаем omega_H = -5e11
    uk, vk, _, _, _, _, om = core.compute_heavy_mode_scalar(
        ki=1e4, thi=0.0, d=1e-5, alpha_ex=1e-11, theta_M=0.0,
        omega_M=1e10, omega_H=-5e11, omega_M_half=5e9, omega_M_sqrt2=7e9
    )

    assert om == 0.0, "При отрицательной дисперсии частота должна обнуляться"
    assert uk == 0.0, "При нестабильности u_k должно быть 0"
    assert vk == 0.0j, "При нестабильности v_k должно быть 0j"


# =====================================================================
# ТЕСТЫ 3: СИММЕТРИИ
# =====================================================================

def test_angular_periodicity(state_in_plane):
    """
    ФИЗИЧЕСКАЯ СИММЕТРИЯ:
    Вектор k, направленный под углом theta, и вектор под углом theta + pi (назад)
    имеют одинаковую частоту в изотропной среде (или симметричной анизотропной).
    """
    k_mag = 1e5
    theta_1 = math.pi / 6  # 30 градусов
    theta_2 = theta_1 + math.pi  # 210 градусов
    theta_3 = theta_1 + 2 * math.pi  # 390 градусов

    _, _, om1 = core.compute_light_mode_scalar(k_mag, theta_1, *state_in_plane.numba_args)
    _, _, om2 = core.compute_light_mode_scalar(k_mag, theta_2, *state_in_plane.numba_args)
    _, _, om3 = core.compute_light_mode_scalar(k_mag, theta_3, *state_in_plane.numba_args)

    np.testing.assert_allclose(om1, om2, atol=1e-8, err_msg="Нарушена симметрия k -> -k (сдвиг на pi)")
    np.testing.assert_allclose(om1, om3, atol=1e-8, err_msg="Нарушена периодичность 2*pi")


def test_out_of_plane_isotropy(state_out_of_plane):
    """Для перпендикулярно намагниченной пленки дисперсия изотропна (не зависит от угла волны)."""
    k_mag = 5e5

    # Считаем для волны, летящей вдоль оси X
    _, _, om_x = core.compute_light_mode_scalar(k_mag, 0.0, *state_out_of_plane.numba_args)

    # Считаем для волны, летящей вдоль оси Y
    _, _, om_y = core.compute_light_mode_scalar(k_mag, math.pi / 2, *state_out_of_plane.numba_args)

    # Считаем для произвольного угла
    _, _, om_rand = core.compute_light_mode_scalar(k_mag, 1.234, *state_out_of_plane.numba_args)

    np.testing.assert_allclose(om_x, om_y, atol=1e-8)
    np.testing.assert_allclose(om_x, om_rand, atol=1e-8)


# =====================================================================
# ТЕСТЫ 4: РЕШАТЕЛИ
# =====================================================================


def test_equilibrium_angle_solver_limits():
    """Тест поиска равновесного угла намагниченности."""

    # 1. При He=0 намагниченность обязана лечь в плоскость (+pi/2 или -pi/2).
    # Проверяем по модулю!
    state_zero_field = core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], 0.0, theta_H_rad=math.pi / 4, d_si=PARAMS_SI['d']
    )
    np.testing.assert_allclose(np.abs(state_zero_field.theta_M), math.pi / 2, atol=1e-5,
                               err_msg="При He=0 намагниченность не легла в плоскость")

    # 2. Бесконечное поле: спины смотрят строго по полю
    target_angle = 1.0
    state_infinite_field = core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], 100.0, theta_H_rad=target_angle, d_si=PARAMS_SI['d']
    )
    np.testing.assert_allclose(state_infinite_field.theta_M, target_angle, atol=1e-3,
                               err_msg="При огромном поле намагниченность не совпала с He")

