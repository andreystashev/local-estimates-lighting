import numpy as np
import matplotlib.pyplot as plt

# Создание сцены
def create_box(size):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    (length, width, height) = size

    ax.plot([0, length], [0, 0], [0, 0], color='k')
    ax.plot([0, 0], [0, width], [0, 0], color='k')
    ax.plot([length, length], [0, width], [0, 0], color='k')
    ax.plot([0, length], [width, width], [0, 0], color='k')
    ax.plot([0, 0], [0, 0], [0, height], color='k')
    ax.plot([length, length], [0, 0], [0, height], color='k')
    ax.plot([0, 0], [width, width], [0, height], color='k')
    ax.plot([length, length], [width, width], [0, height], color='k')

    return ax

# Функция для генерации случайных точек в е-окрестности
def generate_random_points(center, epsilon, num_points):
    points = np.random.normal(loc=center, scale=epsilon, size=(num_points, 3))
    return points

# Функция для вычисления двойной локальной оценки
def double_local_estimate(r, l, r_s_prime, k_function):
    M = np.mean(k_function(r_s_prime, r))
    Q_t = np.zeros(len(r_s_prime))
    for t in range(len(r_s_prime)):
        rs_double_prime = r - np.dot(np.dot((r - r_s_prime[t]), l), l)
        k1_value = k_function(r_s_prime[t], rs_double_prime)
        Q_t[t] = np.prod(k1_value)
    return np.dot(M, np.sum(Q_t))


def kernel_function(r_s_prime, r_s_double_prime):
    return 1 / np.linalg.norm(r_s_prime - r_s_double_prime)

box_size = (5, 5, 5)
num_directions = 10  # Количество направлений визирования
epsilon = 0.1  # Радиус е-окрестности
num_points_per_direction = 10  # Количество случайных точек в е-окрестности

# Создание сцены
ax = create_box(box_size)

# В данном случае генерируются случайные направления
directions = np.random.randn(num_directions, 3)

for direction in directions:
    r = np.random.uniform(size=3) * np.array(box_size)  # Случайная точка в сцене
    r_s_prime = generate_random_points(r, epsilon, num_points_per_direction)  # Случайные точки в е-окрестности
    L = double_local_estimate(r, direction, r_s_prime, kernel_function)  # Вычисление двойной локальной оценки
    ax.scatter(r[0], r[1], r[2], color='red')
    ax.quiver(r[0], r[1], r[2], direction[0], direction[1], direction[2], color='blue')
    ax.text(r[0], r[1], r[2], f'{L:.2f}', color='black')

plt.show()
