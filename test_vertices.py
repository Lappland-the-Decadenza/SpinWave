import pytest
import numpy as np
import math
import timeit

import core
import vertices
import utils


# =====================================================================
# 0. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СТАРЫХ АЛГОРИТМОВ
# =====================================================================

def get_mode_scalars(k_complex, state: core.SystemState):
    """Обертка для удобного вызова скалярного ядра в тестах."""
    return core.compute_heavy_mode_scalar(
        np.abs(k_complex), np.angle(k_complex), *state.numba_args
    )


# =====================================================================
# 1. СТАРЫЕ (НЕОПТИМИЗИРОВАННЫЕ) ФУНКЦИИ ДЛЯ СРАВНЕНИЯ
# =====================================================================

def unoptimized_W_vertex(vectors, state):
    """Старая, прямая реализация W_{12,34} без переиспользования сумм мод."""
    k1, k2, k3, k4 = vectors

    u1, v1, _, b1, q1, _, _ = get_mode_scalars(k1, state)
    u2, v2, _, b2, q2, _, _ = get_mode_scalars(k2, state)
    u3, v3, _, b3, q3, _, _ = get_mode_scalars(k3, state)
    u4, v4, _, b4, q4, _, _ = get_mode_scalars(k4, state)

    v1_c, v2_c = np.conj(v1), np.conj(v2)

    Phi_D = -0.25 * (b1 + b2 + b3)
    Phi_E = -0.25 * (b4 + b1 + b2)
    Phi_F = -0.25 * (b3 + b4 + b1)
    Phi_G = -0.25 * (b2 + b3 + b4)

    q_sum = -0.25 * (q1 + q2 + q3 + q4)

    _, _, _, _, _, g12_plus, _ = get_mode_scalars(k1 + k2, state)
    _, _, _, _, _, g34_plus, _ = get_mode_scalars(k3 + k4, state)
    _, _, _, _, _, g13_minus, _ = get_mode_scalars(k1 - k3, state)
    _, _, _, _, _, g14_minus, _ = get_mode_scalars(k1 - k4, state)
    _, _, _, _, _, g23_minus, _ = get_mode_scalars(k2 - k3, state)
    _, _, _, _, _, g24_minus, _ = get_mode_scalars(k2 - k4, state)

    Psi_A = q_sum + 0.25 * (g13_minus + g14_minus + g23_minus + g24_minus)
    Psi_B = q_sum + 0.25 * (g12_plus + g24_minus + g13_minus + g34_plus)
    Psi_C = q_sum + 0.25 * (g12_plus + g14_minus + g23_minus + g34_plus)

    term1 = Psi_A * (u1 * u2 * u3 * u4 + v1_c * v2_c * v3 * v4)
    term2 = Psi_B * (u1 * v2_c * u3 * v4 + v1_c * u2 * v3 * u4)
    term3 = Psi_C * (u1 * v2_c * v3 * u4 + v1_c * u2 * u3 * v4)
    term4 = Phi_D * (v1_c * v2_c * u3 * v4) + np.conj(Phi_D) * (u1 * u2 * v3 * u4)
    term5 = Phi_E * (v1_c * v2_c * v3 * u4) + np.conj(Phi_E) * (u1 * u2 * u3 * v4)
    term6 = Phi_F * (v1_c * u2 * u3 * u4) + np.conj(Phi_F) * (u1 * v2_c * v3 * v4)
    term7 = Phi_G * (u1 * v2_c * u3 * u4) + np.conj(Phi_G) * (v1_c * u2 * v3 * v4)

    return term1 + term2 + term3 + term4 + term5 + term6 + term7


def unoptimized_T_vertex(vectors, state):
    """Старая реализация T_{12,34} без кэширования."""
    k1, k2, k3, k4 = vectors

    u1, v1, D1, _, _, _, om1 = get_mode_scalars(k1, state)
    u2, v2, D2, _, _, _, om2 = get_mode_scalars(k2, state)
    u3, v3, D3, _, _, _, om3 = get_mode_scalars(k3, state)
    u4, v4, D4, _, _, _, om4 = get_mode_scalars(k4, state)

    u_12p, v_12p, D_12p, _, _, _, om_12p = get_mode_scalars(k1 + k2, state)
    u_34p, v_34p, D_34p, _, _, _, om_34p = get_mode_scalars(k3 + k4, state)
    u_13m, v_13m, D_13m, _, _, _, om_13m = get_mode_scalars(k1 - k3, state)
    u_14m, v_14m, D_14m, _, _, _, om_14m = get_mode_scalars(k1 - k4, state)
    u_23m, v_23m, D_23m, _, _, _, om_23m = get_mode_scalars(k2 - k3, state)
    u_24m, v_24m, D_24m, _, _, _, om_24m = get_mode_scalars(k2 - k4, state)

    U = vertices._three_wave_U_scalar
    V = vertices._three_wave_V_scalar

    n1 = np.conj(U(u1, v1, D1, u2, v2, D2, u_12p, v_12p, D_12p)) * U(u3, v3, D3, u4, v4, D4, u_34p, v_34p, D_34p)
    n2 = np.conj(V(u1, v1, D1, u2, v2, D2, u_12p, v_12p, D_12p)) * V(u3, v3, D3, u4, v4, D4, u_34p, v_34p, D_34p)
    n3 = np.conj(V(u2, v2, D2, u_24m, v_24m, D_24m, u4, v4, D4)) * V(u3, v3, D3, u_13m, v_13m, D_13m, u1, v1, D1)
    n4 = np.conj(V(u1, v1, D1, u_13m, v_13m, D_13m, u3, v3, D3)) * V(u4, v4, D4, u_24m, v_24m, D_24m, u2, v2, D2)
    n5 = np.conj(V(u2, v2, D2, u_23m, v_23m, D_23m, u3, v3, D3)) * V(u4, v4, D4, u_14m, v_14m, D_14m, u1, v1, D1)
    n6 = np.conj(V(u1, v1, D1, u_14m, v_14m, D_14m, u4, v4, D4)) * V(u3, v3, D3, u_23m, v_23m, D_23m, u2, v2, D2)

    d1 = om_12p + om1 + om2
    d2 = om_34p - om3 - om4
    d3 = om_13m + om3 - om1
    d4 = om_24m + om4 - om2
    d5 = om_14m + om4 - om1
    d6 = om_23m + om3 - om2

    total = sum(utils.safe_divide(n, d) for n, d in zip([n1, n2, n3, n4, n5, n6], [d1, d2, d3, d4, d5, d6]))
    return -2.0 * total


def unoptimized_W_tilde(vectors, state):
    """Медленная обертка для расчета через циклы Python."""
    is_scalar = np.ndim(vectors[0]) == 0
    if is_scalar:
        return unoptimized_W_vertex(vectors, state) + unoptimized_T_vertex(vectors, state)
    else:
        k1_arr, k2_arr, k3_arr, k4_arr = vectors
        n = len(k1_arr)
        res = np.empty(n, dtype=np.complex128)
        for i in range(n):
            v = (k1_arr[i], k2_arr[i], k3_arr[i], k4_arr[i])
            res[i] = unoptimized_W_vertex(v, state) + unoptimized_T_vertex(v, state)
        return res


# =====================================================================
# ФИКСТУРЫ (Данные)
# =====================================================================

@pytest.fixture
def mock_state():
    return core.SystemState.from_si(140056.35, 3.603e-12, 0.8, math.pi / 4, 97.0e-9)


@pytest.fixture
def mock_vectors():
    """
    Скалярные векторы для проверки точности физики и симметрий.
    ВАЖНО: Для физических тестов вектора должны соблюдать закон сохранения импульса!
    k1 + k2 = k3 + k4  =>  k4 = k1 + k2 - k3
    """
    k1 = 2e4 + 1j * 5e3
    k2 = -1e4 + 1j * 2e3
    k3 = 1.5e4 - 1j * 3e3
    k4 = k1 + k2 - k3
    return k1, k2, k3, k4


@pytest.fixture
def mock_vectors_array():
    """Векторные массивы для тестирования реальной производительности."""
    N = 1000
    k1 = np.full(N, 2e4 + 1j * 5e3)
    k2 = np.full(N, -1e4 + 1j * 2e3)
    k3 = np.full(N, 1.5e4 - 1j * 3e3)
    k4 = k1 + k2 - k3
    return k1, k2, k3, k4


# =====================================================================
# 1. ТЕСТЫ ФИЗИЧЕСКИХ СИММЕТРИЙ И СВОЙСТВ
# =====================================================================

def test_permutation_symmetry(mock_state, mock_vectors):
    """Бозе-симметрия: перестановка входящих или исходящих частиц не меняет амплитуду."""
    k1, k2, k3, k4 = mock_vectors
    W_base = vertices.calculate_W_tilde((k1, k2, k3, k4), mock_state)
    W_swap_in = vertices.calculate_W_tilde((k2, k1, k3, k4), mock_state)
    W_swap_out = vertices.calculate_W_tilde((k1, k2, k4, k3), mock_state)

    assert np.isclose(W_base, W_swap_in), "Нарушена перестановочная симметрия входящих магнонов!"
    assert np.isclose(W_base, W_swap_out), "Нарушена перестановочная симметрия исходящих магнонов!"


def test_three_wave_symmetry(mock_state, mock_vectors):
    """Проверяем внутреннюю симметрию точных 3-магнонных вершин."""
    k1, k2, k3, _ = mock_vectors

    U_123 = vertices.calculate_three_wave_U(k1, k2, k3, mock_state)
    U_213 = vertices.calculate_three_wave_U(k2, k1, k3, mock_state)

    V_12_3 = vertices.calculate_three_wave_V(k1, k2, k3, mock_state)
    V_21_3 = vertices.calculate_three_wave_V(k2, k1, k3, mock_state)

    assert np.isclose(U_123, U_213), "Нарушена симметрия амплитуды U!"
    assert np.isclose(V_12_3, V_21_3), "Нарушена симметрия амплитуды V!"


def test_bare_W_hermitian_symmetry(mock_state, mock_vectors):
    """Эрмитовость голой вершины W: W_{12,34} == W_{34,12}^*."""
    k1, k2, k3, k4 = mock_vectors

    W_forward = unoptimized_W_vertex((k1, k2, k3, k4), mock_state)
    W_backward = unoptimized_W_vertex((k3, k4, k1, k2), mock_state)

    assert np.isclose(W_forward, np.conj(W_backward)), "Голая вершина W потеряла Эрмитовость!"


def test_forward_scattering_reality(mock_state, mock_vectors):
    """
    Тест диагональных элементов: 1 + 2 -> 1 + 2.
    Диагональные элементы W_tilde_{12,12} ОБЯЗАНЫ быть строго действительными.
    """
    k1, k2, _, _ = mock_vectors
    W_diag = vertices.calculate_W_tilde((k1, k2, k1, k2), mock_state)

    relative_imag = np.abs(np.imag(W_diag) / np.real(W_diag))
    assert relative_imag < 1e-12, f"Слишком большая мнимая часть: {relative_imag}"


def test_time_reversal_symmetry(mock_state, mock_vectors):
    """Симметрия пространственной инверсии: W(k1,k2,k3,k4) == W(-k1,-k2,-k3,-k4)."""
    k1, k2, k3, k4 = mock_vectors
    W_forward = vertices.calculate_W_tilde((k1, k2, k3, k4), mock_state)

    # Разворот вектора на 180 градусов (умножение на -1)
    W_backward = vertices.calculate_W_tilde((-k1, -k2, -k3, -k4), mock_state)

    # УБРАЛИ np.conj(), так как пространственная инверсия не сопрягает амплитуду
    assert np.isclose(W_forward, W_backward, rtol=1e-8), "Нарушена инвариантность к пространственной инверсии!"


# =====================================================================
# 2. МАТЕМАТИЧЕСКИЕ ПРЕДЕЛЫ И ЗАЩИТА ОТ КРАШЕЙ
# =====================================================================

def test_vanishing_magnetization_limit(mock_vectors):
    """Если намагниченность и обмен исчезают (Ms -> 0, A -> 0), 4-магнонное взаимодействие должно исчезать."""
    # Ставим Ms = 1e-12 А/м (вместо 1e-5), чтобы \omega_M стало по-настоящему микроскопическим.
    # При 1e-5 мы получали \omega_M ~ 2.2 рад/с из-за огромного коэффициента \gamma (1.76e7)!
    low_ms_state = core.SystemState.from_si(1e-12, 0.0, 0.8, 0.0, 100e-9)
    W_tilde = vertices.calculate_W_tilde(mock_vectors, low_ms_state)

    assert np.abs(W_tilde) < 1e-5, f"Амплитуда не занулилась при отключении магнетизма: {W_tilde}"


def test_t_vertex_singularity_protection():
    """Проверка утилиты safe_divide на предотвращение краша при резонансах (знаменатель = 0)."""
    res = utils.safe_divide(1.0 + 5j, 0.0)
    assert res == 0.0j, "safe_divide не вернул 0.0j при делении на ноль"


def test_instability_propagation(mock_state):
    """Проверка, что Numba-ядра корректно обрабатывают отрицательную дисперсию (H_e < 0)."""
    broken_args = list(mock_state.numba_args)
    # Насильно прокидываем отрицательное эффективное поле в ядро
    broken_args[4] = -1e12

    # Передаем массивы из одного элемента
    k_arr = np.array([1e4 + 0j])
    res = vertices._calculate_W_tilde_numba_kernel(k_arr, k_arr, k_arr, k_arr, *broken_args)

    # Не должно быть падений с ValueError, и результат должен быть нулевым
    assert res[0] == 0.0j, "Вершина должна обнуляться при физической нестабильности мод (Ak <= 0)"


# =====================================================================
# 3. РЕГРЕССИОННЫЙ ТЕСТ И ПРОИЗВОДИТЕЛЬНОСТЬ
# =====================================================================

def test_optimization_correctness(mock_state, mock_vectors):
    """Сверяем оптимизированную Numba-версию со старой реализацией на Python."""
    res_unopt = unoptimized_W_tilde(mock_vectors, mock_state)
    res_opt = vertices.calculate_W_tilde(mock_vectors, mock_state)

    assert np.isclose(res_unopt, res_opt, rtol=1e-10, atol=1e-15), \
        "Оптимизация вершин сломала математику! Результаты расходятся."


def test_vertices_performance(mock_state, mock_vectors_array):
    """Сравниваем скорость старой и новой реализации на реальных массивах."""
    vectors = mock_vectors_array
    repeats = 100

    # Прогрев Numba компилятора
    vertices.calculate_W_tilde(vectors, mock_state)

    t_unopt = timeit.timeit(
        lambda: unoptimized_W_tilde(vectors, mock_state),
        number=repeats
    )

    t_opt = timeit.timeit(
        lambda: vertices.calculate_W_tilde(vectors, mock_state),
        number=repeats
    )

    print(f"\n[BENCHMARK] Расчет W_tilde (Массивы по {len(vectors[0])} эл., {repeats} прогонов):")
    print(f"Без Numba-ядра (Python loop) : {t_unopt:.4f} сек")
    print(f"Numba-ядро                   : {t_opt:.4f} сек")

    speedup = t_unopt / t_opt
    print(f"-> Ускорение в {speedup:.2f} раз!")

    assert speedup > 2.0, f"Оптимизированная версия не дает выигрыша! Ускорение всего {speedup:.2f}x"