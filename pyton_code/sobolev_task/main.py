import numpy as np
import matplotlib.pyplot as plt


def calculate_illuminance(Phi, N, T, rho, r):
    result = 0
    pi_val = np.pi
    # Вычисляем вес каждого луча Q
    Q = np.ones(N + 1) * rho
    # Вычисляем освещенность для каждого луча и каждого соударения
    for n in range(1, N + 1):
        for i in range(T + 1):
            F = 1 / (pi_val * r)
            result += Phi / (pi_val * N) * Q[i] * F

    illuminance = result
    return illuminance


Phi = 1.0  # Световой поток источника
N = 20  # Количество лучей
T = 5  # Количество соударений
rho = np.ones(N + 1) * 0.5  # Коэффициенты отражения

# Расстояния от источника до плоскости r
r_values = np.linspace(0, 10, 100)  # Генерируем значения расстояний от 1 до 20

# Вычисляем распределение облученности на каждом расстоянии r
illuminance_values = np.zeros_like(r_values)
for i, r in enumerate(r_values):
    illuminance_values[i] = calculate_illuminance(Phi, N, T, rho, r)

plt.plot(r_values, illuminance_values, linewidth=2)
plt.xlabel('Расстояние r')
plt.ylabel('Облученность E(r)')
plt.title('Распределение облученности на плоскости r')
plt.grid(True)
plt.show()

