import math

# Функции системы уравнений
def f1(x, y, z):
    return z

def f2(x, y, z):
    return (math.e ** x + y + z) / 3

# Метод Рунге-Кутты 2-го порядка
def runge_kutta_2(x0, y0, z0, h, x_target):
    x, y, z = x0, y0, z0
    steps = 0  # Счетчик шагов

    while x < x_target:
        if x + h > x_target:
            h = x_target - x

        k1_y = h * f1(x, y, z)
        k1_z = h * f2(x, y, z)

        k2_y = h * f1(x + h, y + k1_y, z + k1_z)
        k2_z = h * f2(x + h, y + k1_y, z + k1_z)

        y += (k1_y + k2_y) / 2
        z += (k1_z + k2_z) / 2
        x += h
        steps += 1

    return y, steps  # Возвращаем последнее значение y и количество шагов

# Двойной пересчет (адаптивный шаг) с фиксацией первого достижения точности
def double_recalculation(method, x0, y0, z0, h, x_target, eps):
    steps_total = 0  # Общее количество шагов
    step_count_at_precision = None  # Запоминаем шаг, на котором впервые достигнута точность

    while True:
        y1, steps1 = method(x0, y0, z0, h, x_target)
        y2, steps2 = method(x0, y0, z0, h / 2, x_target)
        
        steps_total += steps1  # Суммируем шаги
        
        if abs(y1 - y2) <= eps:
            if step_count_at_precision is None:
                step_count_at_precision = steps_total  # Запоминаем первый шаг точности

            print(f"Точность {eps} достигнута на шаге {step_count_at_precision}:")
            print(f"  Итоговое значение y: {y1:.6f}")
            print(f"  Общее количество шагов: {steps_total}")
            print(f"  Использованный шаг: {h:.6f}\n")
            return y1, step_count_at_precision

        h /= 2  # Уменьшаем шаг, если точность не достигнута

# Исходные параметры
x0, y0, z0 = 0.0, 1.0, 1.0
h = 0.1
x_target = 1.0
eps = 10**-6  # Точность 10^-6

# Запуск метода Рунге-Кутты 2-го порядка
double_recalculation(runge_kutta_2, x0, y0, z0, h, x_target, eps)
