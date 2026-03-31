import numpy as np
import matplotlib.pyplot as plt

import core
import contour_logic
import analytics
import utils
import plot_utils  # Наш новый модуль графики


# =====================================================================
# ОРКЕСТРАТОР РАСЧЕТОВ (СКАНИРОВАНИЕ ПО ЧАСТОТЕ)
# =====================================================================
def calculate_for_params(p, frequencies, grid_res=150):
    Ms_si, A_si, d_si = p['Ms'], p['A'], p['d']
    k_in_mag_si = p['k']
    theta_H_rad = np.deg2rad(p['theta_H_deg'])
    theta_k_rad = np.deg2rad(p['theta_k_deg'])

    results = {
        'a_th2': [], 'He': [], 'v_g': [], 'k3x': [], 'k3y': [],
        'K_x': [], 'K_y': [], 'E_mismatch': [], 'frequencies': [],
        'k_in_complex': [], 'k3_complex': [], 'k4_complex': [],
        'is_trivial': [], 'k_span': []
    }

    # tqdm больше не нужен!
    for f in frequencies:
        results['frequencies'].append(f)

        He_opt = float(utils.find_He_for_ghz(
            f_ghz=f, k_si=k_in_mag_si, theta_k_rad=theta_k_rad,
            Ms_si=Ms_si, A_si=A_si, theta_H_rad=theta_H_rad, d_si=d_si
        ))

        state = core.SystemState.from_si(Ms_si, A_si, He_opt, theta_H_rad, d_si)

        k_in_complex = (k_in_mag_si * core.SI_TO_CGS_K) * np.exp(1j * theta_k_rad)

        _, v_g_si, _, _ = analytics.compute_pump_parameters(k_in_complex, state)
        k_span, is_trivial = contour_logic.find_contour_boundaries(k_in_complex, state)
        K_x, K_y, E_mismatch = contour_logic.compute_mismatch_grid(k_in_complex, k_span, state, grid_res)

        a_th2, k3_c, k4_c = analytics.find_minimum_threshold_on_contour(
            K_x, K_y, E_mismatch, k_in_complex, is_trivial, state
        )

        results['He'].append(He_opt)
        results['k_in_complex'].append(k_in_complex)
        results['v_g'].append(v_g_si)
        results['a_th2'].append(a_th2)
        results['k3x'].append(np.real(k3_c) * 100)
        results['k3y'].append(np.imag(k3_c) * 100)
        results['K_x'].append(K_x)
        results['K_y'].append(K_y)
        results['E_mismatch'].append(E_mismatch)
        results['k3_complex'].append(k3_c)
        results['k4_complex'].append(k4_c)
        results['is_trivial'].append(is_trivial)
        results['k_span'].append(k_span)

    return {**p, **results}


# =====================================================================
# ФУНКЦИЯ ОТРИСОВКИ ГЛАВНЫХ ГРАФИКОВ
# =====================================================================
def plot_multiple_results(results_list, frequencies):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for i, res in enumerate(results_list):
        lbl, clr = res['label'], res['color']
        ls, m = res.get('linestyle', '-'), markers[i % len(markers)]

        l_th, = axs[0, 0].plot(frequencies, res['a_th2'], marker=m, color=clr, ls=ls, lw=2, ms=6, label=lbl, picker=5)
        l_th.result_data = res

        l_he, = axs[0, 1].plot(frequencies, res['He'], marker=m, color=clr, ls=ls, lw=2, ms=6, label=lbl, picker=5)
        l_he.result_data = res

        l_vg, = axs[1, 0].plot(frequencies, res['v_g'], marker=m, color=clr, ls=ls, lw=2, ms=6, label=lbl, picker=5)
        l_vg.result_data = res

        l_k3x, = axs[1, 1].plot(frequencies, res['k3x'], marker=m, color=clr, ls=ls, lw=2, ms=6,
                                label=f"{lbl} ($k_{{3x}}$)", picker=5)
        l_k3y, = axs[1, 1].plot(frequencies, res['k3y'], marker=m, color=clr, ls='--', lw=2, ms=6, mfc='white',
                                label=f"{lbl} ($k_{{3y}}$)", picker=5)
        l_k3x.result_data = l_k3y.result_data = res

    axs[0, 0].set(yscale='log', xlabel='Frequency $f$ (GHz)', ylabel=r'$|c_{th}|^2$',
                  title='Square of Critical Amplitude')
    axs[0, 1].set(xlabel='Frequency $f$ (GHz)', ylabel='Required Magnetic Field $H_e$ (Tesla)',
                  title='Resonance Magnetic Field')
    axs[1, 0].set(xlabel='Frequency $f$ (GHz)', ylabel='Group Velocity $v_g$ (m/s)', title='Pump Group Velocity')
    axs[1, 1].set(xlabel='Frequency $f$ (GHz)', ylabel='Wave vector component (1/m)',
                  title=r'Optimal Scattered Vector $\mathbf{k}_3$')

    for ax in axs.flat:
        ax.grid(True, which="both" if ax.get_yscale() == 'log' else "major", linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()

    def on_pick(event):
        artist = event.artist
        if hasattr(artist, 'result_data'):
            # Вызываем универсальную функцию из нового модуля
            plot_utils.show_interactive_contour_popup(artist.result_data, event.ind[0])

    fig.canvas.mpl_connect('pick_event', on_pick)
    print("Графики построены! Кликните на любую точку на графике, чтобы посмотреть карту невязки.")
    plt.show()


if __name__ == "__main__":
    FREQ_MIN, FREQ_MAX, NUM_POINTS = 3.0, 30.0, 30
    frequencies = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)

    params = [
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'k': 3.0e6, 'theta_H_deg': 90.0, 'theta_k_deg': 0.0,
         'label': r'$\theta_k=0^\circ$', 'color': 'blue'},
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'k': 3.0e6, 'theta_H_deg': 90.0, 'theta_k_deg': 90.0,
         'label': r'$\theta_k=90^\circ$', 'color': 'purple'}
    ]

    print("Запуск мгновенного сканирования...")
    all_results = [calculate_for_params(p, frequencies) for p in params]
    plot_multiple_results(all_results, frequencies)