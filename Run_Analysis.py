# %% [markdown]
# # Magnonics Simulation Lab
# Нажмите "Run All" выше, чтобы запустить расчет и построить графики.

# %%
import numpy as np
import matplotlib.pyplot as plt
import plot_thresholds  # Импортируем твой файл как модуль

# 1. Задаем сетку волновых векторов
K_MIN_SI, K_MAX_SI, NUM_POINTS = 2 * np.pi / 500e-6, 2 * np.pi / 5e-6, 40
k_array_si = np.linspace(K_MIN_SI, K_MAX_SI, NUM_POINTS)

# 2. Параметры образцов
params = [
    {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'He': 0.8, 'theta_H_deg': 90.0, 'theta_k_deg': 0.0,
     'label': r'$\theta_k = 0^\circ$', 'color': 'blue'},
    {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'He': 0.8, 'theta_H_deg': 90.0, 'theta_k_deg': 90.0,
     'label': r'$\theta_k = 90^\circ$', 'color': 'red'}
]

print("Запуск расчета... Подождите несколько секунд.")

# 3. Считаем
all_results = [plot_thresholds.calculate_all_vs_k(p, k_array_si) for p in params]

# 4. Рисуем
# В блокнотах графики появятся прямо здесь под ячейкой
plot_thresholds.plot_results(all_results)
