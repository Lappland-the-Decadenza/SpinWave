import pytest
import numpy as np
import math

import core
import utils


# =====================================================================
# ФИКСТУРЫ
# =====================================================================

@pytest.fixture
def state_in_plane():
    """Стандартное состояние пленки (ЖИГ, поле в плоскости 0.8 Тл)."""
    return core.SystemState.from_si(
        Ms_si=140056.35, A_si=3.603e-12, B_ext_tesla=0.8,
        theta_H_rad=math.pi / 2, d_si=97.0e-9
    )


# =====================================================================
# 1. ТЕСТЫ РЕШАТЕЛЕЙ (Solver)
# =====================================================================

def test_unreachable_frequency_solver(state_in_plane):
    """
    ЗАЩИТА РЕШАТЕЛЯ:
    Попытка найти волновой вектор для частоты ниже щели ФМР (0.1 ГГц).
    Оптимизатор должен вернуть NaN, а не зависнуть или выдать мусор.
    """
    impossible_freq = 0.1
    k_res = utils.find_k_for_ghz(impossible_freq, 0.0, state_in_plane)

    assert np.isnan(k_res), f"Решатель нашел k={k_res} для невозможной частоты 0.1 ГГц!"


def test_solver_consistency(state_in_plane):
    """
    САМОСОГЛАСОВАННОСТЬ:
    Проверка обратимости: find_k(f) -> compute_f(k) == f.
    """
    target_f_ghz = 25.0
    theta_k = math.pi / 4

    # 1. Ищем k для заданной частоты
    k_opt = utils.find_k_for_ghz(target_f_ghz, theta_k, state_in_plane)
    assert not np.isnan(k_opt), f"Решатель не смог найти k для {target_f_ghz} ГГц"

    # 2. Считаем частоту обратно через ядро дисперсии
    _, _, om_check = core.compute_light_mode_scalar(k_opt, theta_k, *state_in_plane.numba_args)
    f_check_ghz = om_check / (2 * math.pi * 1e9)

    np.testing.assert_allclose(f_check_ghz, target_f_ghz, rtol=1e-5,
                               err_msg="Решатель вернул k, дающий неверную частоту!")


# =====================================================================
# 2. ТЕСТЫ СЕТОК (Grids)
# =====================================================================

def test_dense_grid_distribution():
    """
    Проверка генератора сеток: сгущение точек к центру (power=3.0).
    """
    center, min_val, max_val = 0.0, -100.0, 100.0
    res = 11  # Центр в индексе 5

    grid = utils.generate_center_dense_grid(center, min_val, max_val, res, power=3.0)

    assert len(grid) == res
    assert np.isclose(grid[5], center), "Центр сетки смещен!"
    assert np.isclose(grid[0], min_val), "Левая граница не совпадает!"
    assert np.isclose(grid[-1], max_val), "Правая граница не совпадает!"

    # Шаг в центре должен быть значительно меньше шага на краях
    center_step = grid[6] - grid[5]
    edge_step = grid[1] - grid[0]
    assert center_step < edge_step, f"Сетка не сгущается! Центр: {center_step}, Край: {edge_step}"


# =====================================================================
# 3. ТЕСТЫ ЗАЩИТНЫХ ФУНКЦИЙ (Safety)
# =====================================================================

def test_safe_divide():
    """
    Проверка защиты от деления на ноль.
    """
    # 1. Обычное деление (комплексные числа)
    num = 10.0 + 5.0j
    den = 2.0 + 0.0j
    assert utils.safe_divide(num, den) == 5.0 + 2.5j

    # 2. Деление на строгий ноль (должно вернуть 0.0j без ошибки)
    assert utils.safe_divide(1.0, 0.0) == 0.0j

    # 3. Деление на очень маленькое число (ниже порога 1e-18)
    assert utils.safe_divide(1.0, 1e-20) == 0.0j

    # 4. Деление нуля на ноль
    assert utils.safe_divide(0.0, 0.0) == 0.0j