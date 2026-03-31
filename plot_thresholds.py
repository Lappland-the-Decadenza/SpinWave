import numpy as np
import matplotlib.pyplot as plt

import core
import contour_logic
import analytics
import plot_utils  # Наш новый модуль графики


# =====================================================================
# ОРКЕСТРАТОР РАСЧЕТОВ (СКАНИРОВАНИЕ ПО ВОЛНОВОМУ ВЕКТОРУ)
# =====================================================================
def calculate_all_vs_k(p, k_array_si, grid_res=150):
    Ms_si, A_si, d_si = p['Ms'], p['A'], p['d']
    He_tesla = p['He']
    th_H_rad = np.deg2rad(p['theta_H_deg'])
    th_k_rad = np.deg2rad(p['theta_k_deg'])

    results = {
        'a_th2': [], 'P_th': [], 'loss': [], 'v_g': [], 'k_in_plot': [],
        'gamma1': [], 'gamma3': [], 'gamma4': [], 'f1': [], 'f3': [], 'f4': [],
        'K_x': [], 'K_y': [], 'E_mismatch': [], 'k_in_complex': [],
        'k3_complex': [], 'k4_complex': [], 'is_trivial': [], 'k_span': []
    }

    state = core.SystemState.from_si(Ms_si, A_si, He_tesla, th_H_rad, d_si)

    # Моментальный расчет без прогресс-баров
    for k_si in k_array_si:
        k_in_complex = (k_si * core.SI_TO_CGS_K) * np.exp(1j * th_k_rad)

        omega_1, v_g_si, gamma_1, loss_db = analytics.compute_pump_parameters(k_in_complex, state)
        k_span, is_trivial = contour_logic.find_contour_boundaries(k_in_complex, state)
        K_x, K_y, E_mismatch = contour_logic.compute_mismatch_grid(k_in_complex, k_span, state, grid_res)

        a_th2, k3_c, k4_c = analytics.find_minimum_threshold_on_contour(
            K_x, K_y, E_mismatch, k_in_complex, is_trivial, state
        )

        P_th = analytics.calculate_threshold_power(a_th2, v_g_si, omega_1, state)
        omega_3, _, gamma_3, _ = analytics.compute_pump_parameters(k3_c, state)
        omega_4, _, gamma_4, _ = analytics.compute_pump_parameters(k4_c, state)

        results['k_in_plot'].append(k_si * 1e-6)
        results['gamma1'].append(gamma_1);
        results['gamma3'].append(gamma_3);
        results['gamma4'].append(gamma_4)
        results['f1'].append(omega_1 / (2 * np.pi * 1e9))
        results['f3'].append(omega_3 / (2 * np.pi * 1e9))
        results['f4'].append(omega_4 / (2 * np.pi * 1e9))
        results['a_th2'].append(a_th2);
        results['P_th'].append(P_th)
        results['loss'].append(loss_db);
        results['v_g'].append(v_g_si)

        results['K_x'].append(K_x);
        results['K_y'].append(K_y);
        results['E_mismatch'].append(E_mismatch)
        results['k_in_complex'].append(k_in_complex);
        results['k3_complex'].append(k3_c);
        results['k4_complex'].append(k4_c)
        results['is_trivial'].append(is_trivial);
        results['k_span'].append(k_span)

    return {**p, **results}


# =====================================================================
# ОСНОВНАЯ ОТРИСОВКА
# =====================================================================
def plot_results(results_list):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    for res in results_list:
        lbl, clr, k = res['label'], res.get('color', 'blue'), res['k_in_plot']

        l_th, = axs[0, 0].plot(k, res['a_th2'], color=clr, lw=2, label=lbl, picker=5)
        l_g, = axs[0, 1].plot(k, res['gamma1'], color=clr, lw=2, label=f"{lbl} (" r"$\Gamma_1$" ")", picker=5)
        l_f, = axs[0, 2].plot(k, res['f1'], color=clr, lw=2, label=f"{lbl} (" r"$\omega_1$" ")", picker=5)
        l_p, = axs[1, 0].plot(k, res['P_th'], color=clr, lw=2, label=lbl, picker=5)
        l_loss, = axs[1, 1].plot(k, res['loss'], color=clr, lw=2, label=lbl, picker=5)
        l_vg, = axs[1, 2].plot(k, res['v_g'], color=clr, lw=2, label=lbl, picker=5)

        # Привязываем данные ко всем линиям (чтобы клик работал везде)
        for line in [l_th, l_g, l_f, l_p, l_loss, l_vg]:
            line.result_data = res

        axs[0, 1].plot(k, res['gamma3'], color=clr, ls='--', lw=2, alpha=0.8, label=r"$\Gamma_3$")
        axs[0, 1].plot(k, res['gamma4'], color=clr, ls=':', lw=2, alpha=0.8, label=r"$\Gamma_4$")
        axs[0, 2].plot(k, res['f3'], color=clr, ls='--', lw=2, alpha=0.8, label=r"$\omega_3$")
        axs[0, 2].plot(k, res['f4'], color=clr, ls=':', lw=2, alpha=0.8, label=r"$\omega_4$")

    x_label = r'Pump wave vector $k_{in}$ (rad/$\mu$m)'
    axs[0, 0].set(yscale='log', xlabel=x_label, ylabel=r'$|c_{th}|^2$', title='Square of Critical Amplitude')
    axs[0, 1].set(xlabel=x_label, ylabel=r'Relaxation rate $\Gamma$ ($s^{-1}$)',
                  title=r'Damping Rates $\Gamma_1, \Gamma_3, \Gamma_4$')
    axs[0, 2].set(xlabel=x_label, ylabel=r'Frequency / $2\pi$ (GHz)',
                  title=r'Frequencies $\omega_1, \omega_3, \omega_4$')
    axs[1, 0].set(yscale='log', xlabel=x_label, ylabel=r'$P_{th}$ (W/m)', title='Threshold Power')
    axs[1, 1].set(xlabel=x_label, ylabel=fr'Loss over {analytics.L_PROP_M * 1e6} $\mu$m (dB)',
                  title='Linear Propagation Loss')
    axs[1, 2].set(xlabel=x_label, ylabel=r'Group Velocity $v_g$ (m/s)', title='Pump Group Velocity')

    for ax in axs.flat:
        ax.grid(True, which="both" if ax.get_yscale() == 'log' else "major", linestyle='--', alpha=0.6)
        ax.legend()

    plt.suptitle("Threshold Analysis vs Pump Wave Vector", fontsize=16, fontweight='bold')
    plt.tight_layout()

    def on_pick(event):
        artist = event.artist
        if hasattr(artist, 'result_data'):
            # Вызываем универсальную функцию из нового модуля
            plot_utils.show_interactive_contour_popup(artist.result_data, event.ind[0])

    fig.canvas.mpl_connect('pick_event', on_pick)
    print("Графики построены! Кликните на любую сплошную линию на графике, чтобы посмотреть карту невязки.")
    plt.show()


if __name__ == "__main__":
    K_MIN_SI, K_MAX_SI, NUM_POINTS = 2 * np.pi / 500e-6, 2 * np.pi / 5e-6, 40
    k_array_si = np.linspace(K_MIN_SI, K_MAX_SI, NUM_POINTS)

    params = [
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'He': 0.8, 'theta_H_deg': 90.0, 'theta_k_deg': 0.0,
         'label': r'$\theta_k = 0^\circ$', 'color': 'blue'},
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'He': 0.8, 'theta_H_deg': 90.0, 'theta_k_deg': 90.0,
         'label': r'$\theta_k = 90^\circ$', 'color': 'red'}
    ]

    print("Запуск мгновенного расчета...")
    all_results = [calculate_all_vs_k(p, k_array_si) for p in params]
    plot_results(all_results)