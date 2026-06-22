import numpy as np
import matplotlib.pyplot as plt

import core
import contour_logic
import analytics
import utils
import plot_utils


# =====================================================================
# ОРКЕСТРАТОР РАСЧЕТОВ (СКАНИРОВАНИЕ ПО ЧАСТОТЕ ДЛЯ РАЗНЫХ k)
# =====================================================================
def calculate_all_vs_f(p, freq_array_ghz, grid_res=1000):
    Ms_si, A_si, d_si = p['Ms'], p['A'], p['d']
    th_H_rad = np.deg2rad(p['theta_H_deg'])
    th_k_rad = np.deg2rad(p['theta_k_deg'])
    k_si = p['k']

    results = {
        'frequencies': [], 'He': [], 'a_th2': [], 'P_th': [], 'loss': [], 'v_g': [],
        'gamma1': [], 'gamma3': [], 'gamma4': [], 'f1': [], 'f3': [], 'f4': [],
        'K_x': [], 'K_y': [], 'E_mismatch': [], 'k_in_complex': [],
        'k3_complex': [], 'k4_complex': [], 'is_trivial': [], 'k_span': [],
        'proj_k3': [], 'proj_k4': [],
        'contour_k3x': [], 'contour_k3y': [], 'contour_P_th': []  # <-- НОВЫЕ КЛЮЧИ
    }

    for f_ghz in freq_array_ghz:
        results['frequencies'].append(f_ghz)

        He_opt = float(utils.find_He_for_ghz(
            f_ghz=f_ghz, k_si=k_si, theta_k_rad=th_k_rad,
            Ms_si=Ms_si, A_si=A_si, theta_H_rad=th_H_rad, d_si=d_si
        ))

        if np.isnan(He_opt):
            results['He'].append(np.nan)
            for key in ['gamma1', 'gamma3', 'gamma4', 'f1', 'f3', 'f4', 'a_th2', 'P_th', 'loss', 'v_g', 'proj_k3',
                        'proj_k4']:
                results[key].append(np.nan)
            for key in ['K_x', 'K_y', 'E_mismatch', 'k_in_complex', 'k3_complex', 'k4_complex', 'k_span', 'contour_k3x',
                        'contour_k3y', 'contour_P_th']:
                results[key].append(None)
            results['is_trivial'].append(True)
            continue

        state = core.SystemState.from_si(Ms_si, A_si, He_opt, th_H_rad, d_si)
        k_in_complex = (k_si * core.SI_TO_CGS_K) * np.exp(1j * th_k_rad)

        omega_1, v_g_si, gamma_1, loss_db = analytics.compute_pump_parameters(k_in_complex, state)
        k_span, is_trivial = contour_logic.find_contour_boundaries(k_in_complex, state)
        K_x, K_y, E_mismatch = contour_logic.compute_mismatch_grid(k_in_complex, k_span, state, grid_res)

        a_th2, k3_c, k4_c, c_k3x, c_k3y, c_a_th2 = analytics.find_minimum_threshold_on_contour(
            K_x, K_y, E_mismatch, k_in_complex, is_trivial, state
        )

        P_th = analytics.calculate_threshold_power(a_th2, v_g_si, omega_1, state)

        # Векторный расчет P_th для всего контура (NumPy автоматически применит скалярную функцию к массиву)
        c_P_th = analytics.calculate_threshold_power(c_a_th2, v_g_si, omega_1, state) if c_a_th2 is not None else None
        omega_3, _, gamma_3, _ = analytics.compute_pump_parameters(k3_c, state)
        omega_4, _, gamma_4, _ = analytics.compute_pump_parameters(k4_c, state)

        # СТРОГАЯ МАТЕМАТИЧЕСКАЯ ПРОЕКЦИЯ НА ВЕКТОР k_in
        k_in_mag = np.abs(k_in_complex)
        if k_in_mag > 0:
            proj_3_cgs = np.real(k3_c * np.conjugate(k_in_complex)) / k_in_mag
            proj_4_cgs = np.real(k4_c * np.conjugate(k_in_complex)) / k_in_mag
        else:
            proj_3_cgs = np.real(k3_c)
            proj_4_cgs = np.real(k4_c)

        # Перевод в СИ
        proj_3_si = proj_3_cgs / core.SI_TO_CGS_K
        proj_4_si = proj_4_cgs / core.SI_TO_CGS_K

        # ЖЕСТКОЕ УДАЛЕНИЕ СЕТОЧНОГО МУСОРА
        # Как видно на графиках, вектора имеют порядок 3*10^6, а шум болтается в 8-м знаке (доли единицы).
        # Округляем значения до десятков (decimals=-1), чтобы срезать шум дискретной сетки оптимизатора
        # и получить чистые прямые линии.
        proj_3_si = np.round(proj_3_si, decimals=-1)
        proj_4_si = np.round(proj_4_si, decimals=-1)

        results['He'].append(He_opt)
        results['gamma1'].append(gamma_1)
        results['gamma3'].append(gamma_3)
        results['gamma4'].append(gamma_4)
        results['f1'].append(omega_1 / (2 * np.pi * 1e9))
        results['f3'].append(omega_3 / (2 * np.pi * 1e9))
        results['f4'].append(omega_4 / (2 * np.pi * 1e9))
        results['a_th2'].append(a_th2)
        results['P_th'].append(P_th)
        results['loss'].append(loss_db)
        results['v_g'].append(v_g_si)

        results['proj_k3'].append(proj_3_si)
        results['proj_k4'].append(proj_4_si)

        results['K_x'].append(K_x)
        results['K_y'].append(K_y)
        results['E_mismatch'].append(E_mismatch)
        results['k_in_complex'].append(k_in_complex)
        results['k3_complex'].append(k3_c)
        results['k4_complex'].append(k4_c)
        results['is_trivial'].append(is_trivial)
        results['k_span'].append(k_span)
        results['contour_k3x'].append(c_k3x)
        results['contour_k3y'].append(c_k3y)
        results['contour_P_th'].append(c_P_th)

    return {**p, **results}


# =====================================================================
# ОСНОВНАЯ ОТРИСОВКА
# =====================================================================
def plot_results(results_list):
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))

    for res in results_list:
        lbl, clr, f = res['label'], res.get('color', 'blue'), res['frequencies']

        l_th, = axs[0, 0].plot(f, res['a_th2'], color=clr, lw=2, label=lbl, picker=5)
        l_g, = axs[0, 1].plot(f, res['gamma1'], color=clr, lw=2, label=f"{lbl} (" r"$\Gamma_1$" ")", picker=5)
        l_he, = axs[0, 2].plot(f, res['He'], color=clr, lw=2, label=f"{lbl}", picker=5)

        # Линии строгой проекции (очищенные от шума)
        l_proj3, = axs[0, 3].plot(f, res['proj_k3'], color=clr, lw=2, label=lbl, picker=5)
        l_proj4, = axs[1, 3].plot(f, res['proj_k4'], color=clr, lw=2, label=lbl, picker=5)

        l_p, = axs[1, 0].plot(f, res['P_th'], color=clr, lw=2, label=lbl, picker=5)
        l_loss, = axs[1, 1].plot(f, res['loss'], color=clr, lw=2, label=lbl, picker=5)
        l_vg, = axs[1, 2].plot(f, res['v_g'], color=clr, lw=2, label=lbl, picker=5)

        for line in [l_th, l_g, l_he, l_p, l_loss, l_vg, l_proj3, l_proj4]:
            line.result_data = res

        axs[0, 1].plot(f, res['gamma3'], color=clr, ls='--', lw=2, alpha=0.8, label=r"$\Gamma_3$")
        axs[0, 1].plot(f, res['gamma4'], color=clr, ls=':', lw=2, alpha=0.8, label=r"$\Gamma_4$")

    x_label = 'Pump Frequency $f$ (GHz)'
    axs[0, 0].set(yscale='log', xlabel=x_label, ylabel=r'$|c_{th}|^2$', title='Square of Critical Amplitude')
    axs[0, 1].set(xlabel=x_label, ylabel=r'Relaxation rate $\Gamma$ ($s^{-1}$)',
                  title=r'Damping Rates $\Gamma_1, \Gamma_3, \Gamma_4$')
    axs[0, 2].set(xlabel=x_label, ylabel=r'Required Magnetic Field $H_e$ (T)', title=r'Resonance Magnetic Field')

    axs[0, 3].set(xlabel=x_label, ylabel=r'Projection ($m^{-1}$)', title=r'Projection of $k_3$ onto $k_{in}$')
    axs[1, 3].set(xlabel=x_label, ylabel=r'Projection ($m^{-1}$)', title=r'Projection of $k_4$ onto $k_{in}$')

    axs[1, 0].set(yscale='log', xlabel=x_label, ylabel=r'$P_{th}$ (W/m)', title='Threshold Power')
    axs[1, 1].set(xlabel=x_label, ylabel=fr'Loss over {analytics.L_PROP_M * 1e6} $\mu$m (dB)',
                  title='Linear Propagation Loss')
    axs[1, 2].set(xlabel=x_label, ylabel=r'Group Velocity $v_g$ (m/s)', title='Pump Group Velocity')

    for ax in axs.flat:
        if hasattr(ax.yaxis.get_major_formatter(), 'set_useOffset'):
            ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.grid(True, which="both" if ax.get_yscale() == 'log' else "major", linestyle='--', alpha=0.6)
        ax.legend()

    plt.suptitle("Threshold Analysis vs Pump Frequency", fontsize=16, fontweight='bold')
    plt.tight_layout()

    def on_pick(event):
        artist = event.artist
        if hasattr(artist, 'result_data'):
            plot_utils.show_interactive_contour_popup(artist.result_data, event.ind[0])

    fig.canvas.mpl_connect('pick_event', on_pick)
    print("Графики построены! Кликните на любую сплошную линию на графике, чтобы посмотреть карту невязки.")
    plt.show()


if __name__ == "__main__":
    FREQ_MIN, FREQ_MAX, NUM_POINTS = 3.0, 30.0, 30
    freq_array_ghz = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)

    params = [
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 10.0e-9, 'k': 3e6, 'theta_H_deg': 0.0, 'theta_k_deg': 30.0},
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 100.0e-9, 'k': 3e6, 'theta_H_deg': 0.0, 'theta_k_deg': 30.0},
        {'Ms': 140056.35, 'A': 3.603e-12, 'd': 1000.0e-9, 'k': 3e6, 'theta_H_deg': 0.0, 'theta_k_deg': 30.0}
    ]

    cmap = plt.get_cmap('tab10')

    for i, p in enumerate(params):
        p['label'] = rf'$d = {p["d"]:.1e}$ m'
        p['color'] = cmap(i % 10)

    print("Запуск сканирования по частоте...")
    all_results = [calculate_all_vs_f(p, freq_array_ghz) for p in params]
    plot_results(all_results)