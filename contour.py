import numpy as np
import matplotlib.pyplot as plt

import core
import contour_logic
import analytics
import plot_utils

# =====================================================================
# БЛОК ВХОДНЫХ ПАРАМЕТРОВ (В СИСТЕМЕ СИ)
# =====================================================================

PARAMS_SI = {
    'Ms': 139.26e3,      # Намагниченность насыщения (А/м)
    'A': 3.7e-12,        # Обменная константа (Дж/м)
    'd': 1.0e-6,         # Толщина пленки (метры)
    'He': 0.8,           # Внешнее магнитное поле (Тл)
    'theta_H_deg': 90.0  # Угол магнитного поля (градусы)
}

K_IN_MAG_SI = 1e3        # Модуль вектора накачки (1/метр)
THETA_IN_DEG = 90.0      # Угол вектора накачки (градусы)

# Вычислительные настройки
GRID_RES_NORMAL = 400    # Разрешение сетки (высокое)
GRID_RES_TRIVIAL = 150   # Сниженное разрешение


def main():
    print("=== STEP 1: INITIALIZING SYSTEM ===")
    state = core.SystemState.from_si(
        PARAMS_SI['Ms'], PARAMS_SI['A'], PARAMS_SI['He'],
        np.deg2rad(PARAMS_SI['theta_H_deg']), PARAMS_SI['d']
    )

    k_in_mag_cgs = K_IN_MAG_SI * core.SI_TO_CGS_K
    k_in_complex = k_in_mag_cgs * np.exp(1j * np.deg2rad(THETA_IN_DEG))

    print("=== STEP 2: COMPUTING GRID AND CONTOUR ===")
    k_span, is_trivial = contour_logic.find_contour_boundaries(k_in_complex, state)

    grid_res = GRID_RES_TRIVIAL if is_trivial else GRID_RES_NORMAL
    K_x, K_y, E_mismatch = contour_logic.compute_mismatch_grid(k_in_complex, k_span, state, grid_res)

    print("=== STEP 3: ANALYZING THRESHOLD ===")
    a_th2, k3_opt, k4_opt = analytics.find_minimum_threshold_on_contour(
        K_x, K_y, E_mismatch, k_in_complex, is_trivial, state
    )

    omega_in, v_g_si, gamma_in, loss_db = analytics.compute_pump_parameters(k_in_complex, state)
    P_th_w_m = analytics.calculate_threshold_power(a_th2, v_g_si, omega_in, state)

    print("=== STEP 4: PLOTTING (SI Scale) ===")
    # Используем графический движок
    fig, ax = plot_utils.create_contour_figure(
        K_x, K_y, E_mismatch, k_span, is_trivial, k_in_complex, k3_opt, k4_opt
    )

    # Вывод результатов в консоль
    if not is_trivial and not np.isnan(P_th_w_m):
        print("=" * 50)
        print("FINAL RESULTS (IN SI UNITS):")
        print(f"Threshold Power P_th = {P_th_w_m:.4e} W/m")
        print(f"Vector k3: k={np.abs(k3_opt) * 1e2:.2e} 1/m, th={np.rad2deg(np.angle(k3_opt)):.1f}°")
        print(f"Vector k4: k={np.abs(k4_opt) * 1e2:.2e} 1/m, th={np.rad2deg(np.angle(k4_opt)):.1f}°")
        print("=" * 50 + "\n")

    plt.show()

if __name__ == "__main__":
    main()