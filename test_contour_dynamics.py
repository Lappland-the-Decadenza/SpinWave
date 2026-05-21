import numpy as np
import matplotlib.pyplot as plt
import core
import contour_logic
import analytics
import utils
import plot_utils

# =====================================================================
# ПАРАМЕТРЫ ТЕСТИРОВАНИЯ (ИЗОЧАСТОТНОЕ СКАНИРОВАНИЕ)
# =====================================================================
TEST_PARAMS_SI = {
    'Ms': 140056.35,
    'A': 3.603e-12,
    'd': 97.0e-9,
    'f_pump_ghz': 10.0,
    'theta_H_deg': 0.0,
    'theta_k_deg': 0.0
}

K_MIN = 1e4  # 1/m
K_MAX = 1e8  # 1/m
SWEEP_STEPS = 200


# =====================================================================
# ЯДРО ДИАГНОСТИКИ
# =====================================================================

class ContourDiagnosticSuite:
    def __init__(self, params):
        self.p = params
        self.th_H = np.deg2rad(params['theta_H_deg'])
        self.th_k = np.deg2rad(params['theta_k_deg'])

    def run_sweep(self, k_array_si):
        print(f"\n[EXECUTE] Starting Iso-Frequency Sweep at {self.p['f_pump_ghz']} GHz ({len(k_array_si)} points)...")
        results = []

        for k_si in k_array_si:
            He_opt = float(utils.find_He_for_ghz(
                f_ghz=self.p['f_pump_ghz'],
                k_si=k_si,
                theta_k_rad=self.th_k,
                Ms_si=self.p['Ms'],
                A_si=self.p['A'],
                theta_H_rad=self.th_H,
                d_si=self.p['d']
            ))

            if np.isnan(He_opt):
                results.append({'k_si': k_si, 'valid_physics': False})
                continue

            state = core.SystemState.from_si(self.p['Ms'], self.p['A'], He_opt, self.th_H, self.p['d'])
            k_cgs = k_si * core.SI_TO_CGS_K
            k_complex = k_cgs * np.exp(1j * self.th_k)

            k_span, is_trivial_math = contour_logic.find_contour_boundaries(k_complex, state)

            grid_res = 100
            test_span = k_span if not is_trivial_math else k_cgs * 0.5
            K_x, K_y, E_mismatch = contour_logic.compute_mismatch_grid(k_complex, test_span, state, grid_res)

            # ВАЖНО: Подключаем физику порогов, а не только голую математику
            a_th2, k3_c, k4_c = analytics.find_minimum_threshold_on_contour(
                K_x, K_y, E_mismatch, k_complex, is_trivial_math, state
            )

            # Критерий физической тривиальности: если k3 упало обратно в k_in
            dist_to_pump = np.abs(k3_c - k_complex)
            is_physically_trivial = (dist_to_pump < np.abs(k_complex) * 1e-3) or np.isnan(a_th2)

            results.append({
                'k_si': k_si,
                'valid_physics': True,
                'He': He_opt,
                'is_trivial_math': is_trivial_math,
                'is_physically_trivial': is_physically_trivial,
                'k_span_cgs': k_span,
                'k3_complex': k3_c,
                'k4_complex': k4_c,
                'K_x': K_x,
                'K_y': K_y,
                'E_grid': E_mismatch,
                'k_complex': k_complex
            })

        return results

    def print_report(self, results):
        print("\n" + "=" * 60)
        print("DIAGNOSTIC REPORT: MATHEMATICS vs PHYSICS")
        print("=" * 60)

        valid_points = [r for r in results if r['valid_physics']]
        if not valid_points:
            print("CRITICAL: No valid physical states found for this frequency.")
            return

        math_contours = [r for r in valid_points if not r['is_trivial_math']]
        real_contours = [r for r in valid_points if not r['is_trivial_math'] and not r['is_physically_trivial']]
        exchange_zones = [r for r in valid_points if not r['is_trivial_math'] and r['is_physically_trivial']]

        print(f"Total points tested: {len(results)}")
        print(f"Valid frequency states: {len(valid_points)}")
        print("-" * 60)
        print(f"Mathematical roots found (is_trivial=False): {len(math_contours)}")
        print(f"[!] FAKE CONTOURS (Exchange Regime / Trivial Physics): {len(exchange_zones)}")
        print(f"[+] REAL PARAMETRIC CONTOURS (k3 != k_in): {len(real_contours)}")

        if exchange_zones:
            k_min = min(c['k_si'] for c in exchange_zones)
            k_max = max(c['k_si'] for c in exchange_zones)
            print(f"\n[!] WARNING: EXCHANGE REGIME DETECTED in k ∈ [{k_min:.2e}, {k_max:.2e}] 1/m.")
            print("    The algorithm found roots (blue zones exist), but scattering is locked to k_in.")

        if real_contours:
            k_min = min(c['k_si'] for c in real_contours)
            k_max = max(c['k_si'] for c in real_contours)
            print(f"\n[+] SUCCESS: Valid parametric generation in k ∈ [{k_min:.2e}, {k_max:.2e}] 1/m.")

    def plot_interactive_results(self, results):
        valid_points = [r for r in results if r['valid_physics']]
        if not valid_points:
            return

        k_vals = [r['k_si'] for r in valid_points]
        span_vals = [r['k_span_cgs'] / core.SI_TO_CGS_K for r in valid_points]

        # Строгая цветовая дифференциация
        colors = []
        for r in valid_points:
            if r['is_trivial_math']:
                colors.append('red')  # Корней нет вообще
            elif r['is_physically_trivial']:
                colors.append('orange')  # Корни есть, но физика запрещает (Exchange)
            else:
                colors.append('green')  # Успех, реальный контур

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(k_vals, span_vals, c=colors, s=40, picker=5, alpha=0.8, edgecolors='black', linewidth=0.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Pump Wavenumber $k_{in}$ (1/m)')
        ax.set_ylabel('Algorithm Boundary Radius $k_{span}$ (1/m)')
        ax.set_title(f"Parametric Contour Map (f = {self.p['f_pump_ghz']} GHz)\n"
                     f"GREEN: Real FMS | ORANGE: Trivial Exchange | RED: No Roots")
        ax.grid(True, which="both", ls="--", alpha=0.5)

        def on_pick(event):
            idx = event.ind[0]
            r = valid_points[idx]

            status = "REAL CONTOUR" if not r['is_physically_trivial'] else "TRIVIAL EXCHANGE"
            if r['is_trivial_math']: status = "NO ROOTS"

            print(f"\n[VIEW] k = {r['k_si']:.2e} 1/m | He = {r['He']:.4f} T | Status: {status}")

            fig_contour, _ = plot_utils.create_contour_figure(
                r['K_x'], r['K_y'], r['E_grid'], r['k_span_cgs'],
                r['is_physically_trivial'], r['k_complex'], r['k3_complex'], r['k4_complex']
            )
            fig_contour.show()

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.tight_layout()
        plt.show()

    def execute_pipeline(self):
        k_array = np.logspace(np.log10(K_MIN), np.log10(K_MAX), SWEEP_STEPS)
        results = self.run_sweep(k_array)
        self.print_report(results)
        self.plot_interactive_results(results)


if __name__ == "__main__":
    tester = ContourDiagnosticSuite(TEST_PARAMS_SI)
    tester.execute_pipeline()