import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def create_contour_figure(K_x, K_y, E_mismatch, k_span, is_trivial, k_in_complex, k3_opt=None, k4_opt=None,
                          custom_title=""):
    """
    Универсальное ядро отрисовки контура.
    Переводит всё в СИ и строит тепловую карту + векторы.
    Возвращает объекты fig, ax.
    """
    # Перевод массивов и векторов из СГС (1/см) в СИ (1/м)
    K_x_SI, K_y_SI = K_x * 1e2, K_y * 1e2
    k_span_SI = k_span * 1e2
    kx_in_SI, ky_in_SI = np.real(k_in_complex) * 1e2, np.imag(k_in_complex) * 1e2
    P_tot_x_SI, P_tot_y_SI = 2.0 * kx_in_SI, 2.0 * ky_in_SI

    fig, ax = plt.subplots(figsize=(11, 8))

    # Настройка симметричного логарифмического градиента
    max_err = max(abs(np.nanmin(E_mismatch)), abs(np.nanmax(E_mismatch)))
    if np.isnan(max_err) or max_err == 0:
        max_err = 1e8

    heatmap_norm = SymLogNorm(linthresh=5e7, vmin=-max_err, vmax=max_err, base=10)

    cmap_plot = ax.pcolormesh(K_x_SI, K_y_SI, E_mismatch, shading='nearest', cmap='RdBu_r', norm=heatmap_norm,
                              alpha=0.9)
    cbar = fig.colorbar(cmap_plot, ax=ax)
    cbar.set_label(r'Energy mismatch $\Delta \omega$ (Symmetrical Log)')

    if not is_trivial and k3_opt is not None and k4_opt is not None:
        ax.contour(K_x_SI, K_y_SI, E_mismatch, levels=[0.0], colors='black', linewidths=3)
        title = custom_title if custom_title else r'Resonance Contour and Optimal Vectors'
        ax.set_title(title, fontweight='bold', fontsize=14)

        opt_k3x_SI, opt_k3y_SI = np.real(k3_opt) * 1e2, np.imag(k3_opt) * 1e2
        opt_k4x_SI, opt_k4y_SI = np.real(k4_opt) * 1e2, np.imag(k4_opt) * 1e2

        ax.quiver(0, 0, kx_in_SI, ky_in_SI, color='black', angles='xy', scale_units='xy', scale=1, width=0.008,
                  label=r'Pump $\mathbf{k}_{in}$')
        ax.quiver(0, 0, opt_k3x_SI, opt_k3y_SI, color='cyan', edgecolor='black', linewidth=1, angles='xy',
                  scale_units='xy', scale=1, width=0.008, label=r'Opt. $\mathbf{k}_3$')
        ax.quiver(opt_k3x_SI, opt_k3y_SI, opt_k4x_SI, opt_k4y_SI, color='lime', edgecolor='black', linewidth=1,
                  angles='xy', scale_units='xy', scale=1, width=0.008, label=r'Opt. $\mathbf{k}_4$')
        ax.plot(P_tot_x_SI, P_tot_y_SI, 'ko', markersize=8, label=r'$2\mathbf{k}_{in}$')
    else:
        title = custom_title if custom_title else 'EXCHANGE REGIME\n(Only trivial scattering allowed)'
        ax.set_title(title, color='darkred', fontweight='bold')
        ax.plot(kx_in_SI, ky_in_SI, 'w*', markersize=20, markeredgecolor='black',
                label=r'Trivial point $\mathbf{k}_3 = \mathbf{k}_{in}$')
        ax.quiver(0, 0, kx_in_SI, ky_in_SI, color='black', angles='xy', scale_units='xy', scale=1, width=0.008,
                  label=r'Pump $\mathbf{k}_{in}$')
        ax.plot(P_tot_x_SI, P_tot_y_SI, 'ko', markersize=8, label=r'$2\mathbf{k}_{in}$')

    ax.set_xlim(kx_in_SI - k_span_SI, kx_in_SI + k_span_SI)
    ax.set_ylim(ky_in_SI - k_span_SI, ky_in_SI + k_span_SI)
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$ (1/m)', fontsize=12)
    ax.set_ylabel('$k_y$ (1/m)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()

    return fig, ax


def show_interactive_contour_popup(res_data, idx):
    """Обертка для отображения всплывающего окна по клику из plot_W и thresholds."""
    K_x = res_data['K_x'][idx]
    K_y = res_data['K_y'][idx]
    E_mismatch = res_data['E_mismatch'][idx]
    k_in_complex = res_data['k_in_complex'][idx]
    k3_c = res_data['k3_complex'][idx]
    k4_c = res_data['k4_complex'][idx]
    is_trivial = res_data['is_trivial'][idx]
    k_span = res_data['k_span'][idx]

    label = res_data.get('label', '')
    freq_ghz = res_data.get('frequencies', res_data.get('f1'))[idx]

    if not is_trivial:
        title = f'[{label}] Resonance Contour (f = {freq_ghz:.2f} GHz)'
    else:
        title = f'[{label}] EXCHANGE REGIME\n(Only trivial scattering allowed, f = {freq_ghz:.2f} GHz)'

    # Вызываем наше единое графическое ядро
    fig, ax = create_contour_figure(
        K_x, K_y, E_mismatch, k_span, is_trivial,
        k_in_complex, k3_c, k4_c, custom_title=title
    )
    fig.show()