import numpy as np
from numba import njit
import core
import utils


# =====================================================================
# ВЫЧИСЛИТЕЛЬНОЕ ЯДРО NUMBA (Скалярные операции)
# =====================================================================


@njit(cache=True)
def _L_factor_scalar(u, v, D):
    return D * u + np.conj(D) * v


@njit(cache=True)
def _three_wave_U_scalar(u1, v1, D1, u2, v2, D2, u3, v3, D3):
    L1 = _L_factor_scalar(u1, v1, D1)
    L2 = _L_factor_scalar(u2, v2, D2)
    L3 = _L_factor_scalar(u3, v3, D3)

    term1 = L1 * (u2 * v3 + v2 * u3)
    term2 = L2 * (u1 * v3 + v1 * u3)
    term3 = L3 * (u1 * v2 + v1 * u2)
    return -0.5 * (term1 + term2 + term3)


@njit(cache=True)
def _three_wave_V_scalar(u1, v1, D1, u2, v2, D2, u3, v3, D3):
    L1 = _L_factor_scalar(u1, v1, D1)
    L2 = _L_factor_scalar(u2, v2, D2)
    L3_conj = np.conj(_L_factor_scalar(u3, v3, D3))
    v3_c = np.conj(v3)

    term1 = L1 * (u2 * u3 + v2 * v3_c)
    term2 = L2 * (u1 * u3 + v1 * v3_c)
    term3 = L3_conj * (u1 * v2 + v1 * u2)
    return -0.5 * (term1 + term2 + term3)


@njit(cache=True)
def _calculate_W_vertex_numba(
        u1, u2, u3, u4, v1, v2, v3, v4, b1, b2, b3, b4, q1, q2, q3, q4,
        g_12p, g_34p, g_13m, g_14m, g_23m, g_24m
):
    v1_c = np.conj(v1)
    v2_c = np.conj(v2)

    Phi_D = -0.25 * (b1 + b2 + b3)
    Phi_E = -0.25 * (b4 + b1 + b2)
    Phi_F = -0.25 * (b3 + b4 + b1)
    Phi_G = -0.25 * (b2 + b3 + b4)

    Phi_D_c = np.conj(Phi_D)
    Phi_E_c = np.conj(Phi_E)
    Phi_F_c = np.conj(Phi_F)
    Phi_G_c = np.conj(Phi_G)

    q_sum = -0.25 * (q1 + q2 + q3 + q4)

    Psi_A = q_sum + 0.25 * (g_13m + g_14m + g_23m + g_24m)
    Psi_B = q_sum + 0.25 * (g_12p + g_24m + g_13m + g_34p)
    Psi_C = q_sum + 0.25 * (g_12p + g_14m + g_23m + g_34p)

    term1 = Psi_A * (u1 * u2 * u3 * u4 + v1_c * v2_c * v3 * v4)
    term2 = Psi_B * (u1 * v2_c * u3 * v4 + v1_c * u2 * v3 * u4)
    term3 = Psi_C * (u1 * v2_c * v3 * u4 + v1_c * u2 * u3 * v4)

    term4 = Phi_D * (v1_c * v2_c * u3 * v4) + Phi_D_c * (u1 * u2 * v3 * u4)
    term5 = Phi_E * (v1_c * v2_c * v3 * u4) + Phi_E_c * (u1 * u2 * u3 * v4)
    term6 = Phi_F * (v1_c * u2 * u3 * u4) + Phi_F_c * (u1 * v2_c * v3 * v4)
    term7 = Phi_G * (u1 * v2_c * u3 * u4) + Phi_G_c * (v1_c * u2 * v3 * v4)

    return term1 + term2 + term3 + term4 + term5 + term6 + term7


@njit(cache=True)
def _calculate_T_vertex_numba(
        u1, u2, u3, u4, v1, v2, v3, v4, D1, D2, D3, D4, om1, om2, om3, om4,
        u_12p, v_12p, D_12p, om_12p, u_34p, v_34p, D_34p, om_34p,
        u_13m, v_13m, D_13m, om_13m, u_14m, v_14m, D_14m, om_14m,
        u_23m, v_23m, D_23m, om_23m, u_24m, v_24m, D_24m, om_24m
):
    n1 = np.conj(_three_wave_U_scalar(u1, v1, D1, u2, v2, D2, u_12p, v_12p, D_12p)) * _three_wave_U_scalar(u3, v3, D3,
                                                                                                           u4, v4, D4,
                                                                                                           u_34p, v_34p,
                                                                                                           D_34p)
    n2 = np.conj(_three_wave_V_scalar(u1, v1, D1, u2, v2, D2, u_12p, v_12p, D_12p)) * _three_wave_V_scalar(u3, v3, D3,
                                                                                                           u4, v4, D4,
                                                                                                           u_34p, v_34p,
                                                                                                           D_34p)
    n3 = np.conj(_three_wave_V_scalar(u2, v2, D2, u_24m, v_24m, D_24m, u4, v4, D4)) * _three_wave_V_scalar(u3, v3, D3,
                                                                                                           u_13m, v_13m,
                                                                                                           D_13m, u1,
                                                                                                           v1, D1)
    n4 = np.conj(_three_wave_V_scalar(u1, v1, D1, u_13m, v_13m, D_13m, u3, v3, D3)) * _three_wave_V_scalar(u4, v4, D4,
                                                                                                           u_24m, v_24m,
                                                                                                           D_24m, u2,
                                                                                                           v2, D2)
    n5 = np.conj(_three_wave_V_scalar(u2, v2, D2, u_23m, v_23m, D_23m, u3, v3, D3)) * _three_wave_V_scalar(u4, v4, D4,
                                                                                                           u_14m, v_14m,
                                                                                                           D_14m, u1,
                                                                                                           v1, D1)
    n6 = np.conj(_three_wave_V_scalar(u1, v1, D1, u_14m, v_14m, D_14m, u4, v4, D4)) * _three_wave_V_scalar(u3, v3, D3,
                                                                                                           u_23m, v_23m,
                                                                                                           D_23m, u2,
                                                                                                           v2, D2)

    d1 = om_12p + om1 + om2
    d2 = om_34p - om3 - om4
    d3 = om_13m + om3 - om1
    d4 = om_24m + om4 - om2
    d5 = om_14m + om4 - om1
    d6 = om_23m + om3 - om2

    t1 = utils.safe_divide(n1, d1)
    t2 = utils.safe_divide(n2, d2)
    t3 = utils.safe_divide(n3, d3)
    t4 = utils.safe_divide(n4, d4)
    t5 = utils.safe_divide(n5, d5)
    t6 = utils.safe_divide(n6, d6)

    return -2.0 * (t1 + t2 + t3 + t4 + t5 + t6)


# =====================================================================
# ГЛАВНЫЕ ЯДРА (Итераторы)
# =====================================================================

@njit(cache=True)
def _calculate_W_tilde_numba_kernel(k1_arr, k2_arr, k3_arr, k4_arr,
                                    d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    n = len(k1_arr)
    res = np.empty(n, dtype=np.complex128)

    for i in range(n):
        k1 = k1_arr[i]
        k2 = k2_arr[i]
        k3 = k3_arr[i]
        k4 = k4_arr[i]

        u1, v1, D1, b1, q1, _, om1 = core.compute_heavy_mode_scalar(np.abs(k1), np.angle(k1), d, alpha_ex, theta_M,
                                                                    omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u2, v2, D2, b2, q2, _, om2 = core.compute_heavy_mode_scalar(np.abs(k2), np.angle(k2), d, alpha_ex, theta_M,
                                                                    omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u3, v3, D3, b3, q3, _, om3 = core.compute_heavy_mode_scalar(np.abs(k3), np.angle(k3), d, alpha_ex, theta_M,
                                                                    omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u4, v4, D4, b4, q4, _, om4 = core.compute_heavy_mode_scalar(np.abs(k4), np.angle(k4), d, alpha_ex, theta_M,
                                                                    omega_M, omega_H, omega_M_half, omega_M_sqrt2)

        k12p = k1 + k2
        u_12p, v_12p, D_12p, _, _, g_12p, om_12p = core.compute_heavy_mode_scalar(np.abs(k12p), np.angle(k12p), d,
                                                                                  alpha_ex, theta_M, omega_M, omega_H,
                                                                                  omega_M_half, omega_M_sqrt2)

        k34p = k3 + k4
        u_34p, v_34p, D_34p, _, _, g_34p, om_34p = core.compute_heavy_mode_scalar(np.abs(k34p), np.angle(k34p), d,
                                                                                  alpha_ex, theta_M, omega_M, omega_H,
                                                                                  omega_M_half, omega_M_sqrt2)

        k13m = k1 - k3
        u_13m, v_13m, D_13m, _, _, g_13m, om_13m = core.compute_heavy_mode_scalar(np.abs(k13m), np.angle(k13m), d,
                                                                                  alpha_ex, theta_M, omega_M, omega_H,
                                                                                  omega_M_half, omega_M_sqrt2)

        k14m = k1 - k4
        u_14m, v_14m, D_14m, _, _, g_14m, om_14m = core.compute_heavy_mode_scalar(np.abs(k14m), np.angle(k14m), d,
                                                                                  alpha_ex, theta_M, omega_M, omega_H,
                                                                                  omega_M_half, omega_M_sqrt2)

        k23m = k2 - k3
        u_23m, v_23m, D_23m, _, _, g_23m, om_23m = core.compute_heavy_mode_scalar(np.abs(k23m), np.angle(k23m), d,
                                                                                  alpha_ex, theta_M, omega_M, omega_H,
                                                                                  omega_M_half, omega_M_sqrt2)

        k24m = k2 - k4
        u_24m, v_24m, D_24m, _, _, g_24m, om_24m = core.compute_heavy_mode_scalar(np.abs(k24m), np.angle(k24m), d,
                                                                                  alpha_ex, theta_M, omega_M, omega_H,
                                                                                  omega_M_half, omega_M_sqrt2)

        W = _calculate_W_vertex_numba(
            u1, u2, u3, u4, v1, v2, v3, v4,
            b1, b2, b3, b4, q1, q2, q3, q4,
            g_12p, g_34p, g_13m, g_14m, g_23m, g_24m
        )

        T = _calculate_T_vertex_numba(
            u1, u2, u3, u4, v1, v2, v3, v4, D1, D2, D3, D4, om1, om2, om3, om4,
            u_12p, v_12p, D_12p, om_12p,
            u_34p, v_34p, D_34p, om_34p,
            u_13m, v_13m, D_13m, om_13m,
            u_14m, v_14m, D_14m, om_14m,
            u_23m, v_23m, D_23m, om_23m,
            u_24m, v_24m, D_24m, om_24m
        )

        res[i] = W + T

    return res


@njit(cache=True)
def _calculate_three_wave_U_kernel(k1_arr, k2_arr, k3_arr,
                                   d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    n = len(k1_arr)
    res = np.empty(n, dtype=np.complex128)
    for i in range(n):
        u1, v1, D1, _, _, _, _ = core.compute_heavy_mode_scalar(np.abs(k1_arr[i]), np.angle(k1_arr[i]), d, alpha_ex,
                                                                theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u2, v2, D2, _, _, _, _ = core.compute_heavy_mode_scalar(np.abs(k2_arr[i]), np.angle(k2_arr[i]), d, alpha_ex,
                                                                theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u3, v3, D3, _, _, _, _ = core.compute_heavy_mode_scalar(np.abs(k3_arr[i]), np.angle(k3_arr[i]), d, alpha_ex,
                                                                theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        res[i] = _three_wave_U_scalar(u1, v1, D1, u2, v2, D2, u3, v3, D3)
    return res


@njit(cache=True)
def _calculate_three_wave_V_kernel(k1_arr, k2_arr, k3_arr,
                                   d, alpha_ex, theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2):
    n = len(k1_arr)
    res = np.empty(n, dtype=np.complex128)
    for i in range(n):
        u1, v1, D1, _, _, _, _ = core.compute_heavy_mode_scalar(np.abs(k1_arr[i]), np.angle(k1_arr[i]), d, alpha_ex,
                                                                theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u2, v2, D2, _, _, _, _ = core.compute_heavy_mode_scalar(np.abs(k2_arr[i]), np.angle(k2_arr[i]), d, alpha_ex,
                                                                theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        u3, v3, D3, _, _, _, _ = core.compute_heavy_mode_scalar(np.abs(k3_arr[i]), np.angle(k3_arr[i]), d, alpha_ex,
                                                                theta_M, omega_M, omega_H, omega_M_half, omega_M_sqrt2)
        res[i] = _three_wave_V_scalar(u1, v1, D1, u2, v2, D2, u3, v3, D3)
    return res


# =====================================================================
# ОРКЕСТРАТОРЫ (Слой Python)
# =====================================================================

def calculate_three_wave_U(k1: complex, k2: complex, k3: complex, state: core.SystemState):
    """Амплитуда U_{123}."""
    is_scalar = np.ndim(k1) == 0
    k1_arr, k2_arr, k3_arr = [np.atleast_1d(v).astype(np.complex128) for v in (k1, k2, k3)]

    res = _calculate_three_wave_U_kernel(k1_arr, k2_arr, k3_arr, *state.numba_args)
    return res[0] if is_scalar else res


def calculate_three_wave_V(k1: complex, k2: complex, k3: complex, state: core.SystemState):
    """Амплитуда V_{12,3}."""
    is_scalar = np.ndim(k1) == 0
    k1_arr, k2_arr, k3_arr = [np.atleast_1d(v).astype(np.complex128) for v in (k1, k2, k3)]

    res = _calculate_three_wave_V_kernel(k1_arr, k2_arr, k3_arr, *state.numba_args)
    return res[0] if is_scalar else res


def calculate_W_tilde(vectors: tuple[complex, complex, complex, complex], state: core.SystemState):
    """Полная эффективная амплитуда 4-магнонного взаимодействия W_tilde = W + T."""
    is_scalar = np.ndim(vectors[0]) == 0
    k1_arr, k2_arr, k3_arr, k4_arr = [np.atleast_1d(v).astype(np.complex128) for v in vectors]

    res = _calculate_W_tilde_numba_kernel(k1_arr, k2_arr, k3_arr, k4_arr, *state.numba_args)

    return res[0] if is_scalar else res