# %% [markdown]
# # Magnonics Simulation Lab
# Нажмите "Run All" выше, чтобы запустить расчет и построить графики.

# %%
import numpy as np
import matplotlib.pyplot as plt
import plot_thresholds  # Импортируем твой файл как модуль

# 1. Задаем сетку частот
FREQ_MIN, FREQ_MAX, NUM_POINTS = 3.0, 30.0, 30
freq_array_ghz = np.linspace(FREQ_MIN, FREQ_MAX, NUM_POINTS)

# 2. Параметры образцов (разные значения волнового вектора k)
params = [
    {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'k': 2 * np.pi / 500e-6, 'theta_H_deg': 0.0, 'theta_k_deg': 0.0,
     'label': r'$k = 2 \times 10^6$ m$^{-1}$', 'color': 'blue'},
    {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'k': 2 * np.pi / 50e-6, 'theta_H_deg': 0.0, 'theta_k_deg': 0.0,
     'label': r'$k = 5 \times 10^6$ m$^{-1}$', 'color': 'red'},
    {'Ms': 140056.35, 'A': 3.603e-12, 'd': 97.0e-9, 'k': 2 * np.pi / 5e-6, 'theta_H_deg': 0.0, 'theta_k_deg': 0.0,
     'label': r'$k = 10 \times 10^6$ m$^{-1}$', 'color': 'green'}
]

print("Запуск сканирования по частоте... Подождите несколько секунд.")

# 3. Считаем (теперь вызываем новую функцию calculate_all_vs_f)
all_results = [plot_thresholds.calculate_all_vs_f(p, freq_array_ghz) for p in params]

# 4. Рисуем
# В блокнотах графики появятся прямо здесь под ячейкой
plot_thresholds.plot_results(all_results)