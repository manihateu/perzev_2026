"""
Модуль для проведения численных экспериментов по умножению матриц
"""

import numpy as np
import time
import csv
import json
from typing import List, Dict, Tuple
from matrix_multiply import (
    fill_matrix_divide_conquer,
    multiply_matrices_parallel,
    multiply_matrices_sequential,
    multiply_matrices_linearized_v1,
    multiply_matrices_linearized_v2,
    multiply_matrices_linearized_v3,
    verify_result
)
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def measure_time(func, *args, **kwargs) -> Tuple[float, any]:
    """
    Измерение времени выполнения функции
    
    Returns:
        (время в секундах, результат функции)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start), result


def experiment_threads_dependency():
    """
    Эксперимент 1: Зависимость скорости от количества потоков
    """
    print("=" * 80)
    print("Эксперимент 1: Зависимость скорости от количества потоков")
    print("=" * 80)
    print()
    
    # Размеры матриц для тестирования (от 10 до 10000 с шагом)
    # Для больших размеров (>10000) эксперименты могут занять много времени
    sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    max_threads = cpu_count()
    thread_counts = [1, 2, 4, 8]
    if max_threads > 8:
        thread_counts.append(max_threads)
    
    results = []
    
    for size in sizes:
        print(f"\nРазмер матрицы: {size} x {size}")
        print("-" * 80)
        
        # Генерируем матрицы
        print("Генерация матриц...")
        A = fill_matrix_divide_conquer(size)
        B = fill_matrix_divide_conquer(size, seed=100)
        
        baseline_time = None
        
        for num_threads in thread_counts:
            if num_threads > size:
                continue
            
            # Выполняем несколько прогонов для усреднения
            times = []
            for run in range(3):
                elapsed, C = measure_time(
                    multiply_matrices_parallel, A, B, num_threads
                )
                times.append(elapsed)
                
                # Проверяем корректность
                if not verify_result(A, B, C):
                    print(f"  ОШИБКА: Неверный результат для {num_threads} потоков!")
            
            avg_time = np.mean(times)
            
            if num_threads == 1:
                baseline_time = avg_time
            
            speedup = baseline_time / avg_time if baseline_time else 1.0
            
            print(f"  {num_threads:2d} потоков: {avg_time:10.6f} с, "
                  f"ускорение: {speedup:6.2f}x")
            
            results.append({
                'size': size,
                'threads': num_threads,
                'time': avg_time,
                'speedup': speedup
            })
    
    # Сохраняем результаты
    with open('results_threads.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'threads', 'time', 'speedup'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nРезультаты сохранены в results_threads.csv")
    return results


def experiment_size_dependency():
    """
    Эксперимент 2: Зависимость скорости от размерности матрицы
    """
    print("\n\n" + "=" * 80)
    print("Эксперимент 2: Зависимость скорости от размерности матрицы")
    print("=" * 80)
    print()
    
    # Размеры матриц от 10 до 10000 (с шагом)
    # Можно расширить до 100000, но это займет очень много времени
    sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    num_threads = cpu_count()
    
    results = []
    
    for size in sizes:
        print(f"Размер матрицы: {size} x {size}")
        
        # Генерируем матрицы
        A = fill_matrix_divide_conquer(size)
        B = fill_matrix_divide_conquer(size, seed=100)
        
        # Последовательная версия
        elapsed_seq, C_seq = measure_time(multiply_matrices_sequential, A, B)
        if not verify_result(A, B, C_seq):
            print("  ОШИБКА: Неверный результат последовательной версии!")
        
        # Параллельная версия
        elapsed_par, C_par = measure_time(
            multiply_matrices_parallel, A, B, num_threads
        )
        if not verify_result(A, B, C_par):
            print("  ОШИБКА: Неверный результат параллельной версии!")
        
        speedup = elapsed_seq / elapsed_par
        
        print(f"  Последовательная: {elapsed_seq:10.6f} с")
        print(f"  Параллельная ({num_threads} потоков): {elapsed_par:10.6f} с, "
              f"ускорение: {speedup:6.2f}x")
        print()
        
        results.append({
            'size': size,
            'sequential_time': elapsed_seq,
            'parallel_time': elapsed_par,
            'speedup': speedup,
            'threads': num_threads
        })
    
    # Сохраняем результаты
    with open('results_size.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'sequential_time', 
                                                'parallel_time', 'speedup', 'threads'])
        writer.writeheader()
        writer.writerows(results)
    
    print("Результаты сохранены в results_size.csv")
    return results


def experiment_linearization():
    """
    Эксперимент 3: Сравнение вариантов линеаризации циклов
    """
    print("\n\n" + "=" * 80)
    print("Эксперимент 3: Сравнение вариантов линеаризации циклов")
    print("=" * 80)
    print()
    
    sizes = [10, 50, 100, 200, 500, 1000]
    variants = {
        'Стандартный': multiply_matrices_sequential,
        'Линеаризация v1 (i,k,j)': multiply_matrices_linearized_v1,
        'Линеаризация v2 (k,i,j)': multiply_matrices_linearized_v2,
        'Линеаризация v3 (j,i,k)': multiply_matrices_linearized_v3,
    }
    
    results = []
    
    for size in sizes:
        print(f"\nРазмер матрицы: {size} x {size}")
        print("-" * 80)
        
        A = fill_matrix_divide_conquer(size)
        B = fill_matrix_divide_conquer(size, seed=100)
        
        variant_times = {}
        
        for variant_name, variant_func in variants.items():
            times = []
            for run in range(3):
                elapsed, C = measure_time(variant_func, A, B)
                times.append(elapsed)
                
                if not verify_result(A, B, C):
                    print(f"  ОШИБКА: {variant_name} - неверный результат!")
            
            avg_time = np.mean(times)
            variant_times[variant_name] = avg_time
            
            print(f"  {variant_name:30s}: {avg_time:10.6f} с")
            
            results.append({
                'size': size,
                'variant': variant_name,
                'time': avg_time
            })
        
        # Находим оптимальный вариант
        best = min(variant_times.items(), key=lambda x: x[1])
        print(f"  Лучший вариант: {best[0]} ({best[1]:.6f} с)")
    
    # Сохраняем результаты
    with open('results_linearization.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'variant', 'time'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nРезультаты сохранены в results_linearization.csv")
    return results


def derive_formula(results_threads: List[Dict], results_size: List[Dict]):
    """
    Вывод формулы для оценки скорости выполнения
    
    Пытаемся найти зависимость: T(n, p) = f(n, p)
    где n - размерность матрицы, p - количество потоков
    """
    print("\n\n" + "=" * 80)
    print("Вывод формулы для оценки скорости выполнения")
    print("=" * 80)
    print()
    
    # Подготовка данных
    sizes = sorted(set(r['size'] for r in results_threads))
    threads = sorted(set(r['threads'] for r in results_threads))
    
    # Создаем матрицу данных: время[sizes][threads]
    time_matrix = {}
    for r in results_threads:
        size = r['size']
        thread = r['threads']
        if size not in time_matrix:
            time_matrix[size] = {}
        time_matrix[size][thread] = r['time']
    
    # Попытка найти формулу T(n, p) = a * n^b / p^c
    # Логарифмируем: log(T) = log(a) + b*log(n) - c*log(p)
    
    print("Попытка аппроксимации формулой: T(n, p) = a * n^b / p^c")
    print()
    
    # Подготовка данных для регрессии
    n_values = []
    p_values = []
    t_values = []
    
    for size in sizes:
        for thread in threads:
            if size in time_matrix and thread in time_matrix[size]:
                n_values.append(size)
                p_values.append(thread)
                t_values.append(time_matrix[size][thread])
    
    n_arr = np.array(n_values)
    p_arr = np.array(p_values)
    t_arr = np.array(t_values)
    
    # Логарифмическая регрессия
    log_n = np.log(n_arr)
    log_p = np.log(p_arr)
    log_t = np.log(t_arr)
    
    # Линейная регрессия: log(T) = log(a) + b*log(n) - c*log(p)
    # Используем метод наименьших квадратов
    A = np.vstack([np.ones(len(log_n)), log_n, -log_p]).T
    coeffs = np.linalg.lstsq(A, log_t, rcond=None)[0]
    
    log_a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    a = np.exp(log_a)
    
    print(f"Найденные коэффициенты:")
    print(f"  a = {a:.6e}")
    print(f"  b = {b:.4f}")
    print(f"  c = {c:.4f}")
    print()
    print(f"Формула: T(n, p) = {a:.6e} * n^{b:.4f} / p^{c:.4f}")
    print()
    
    # Вычисляем R²
    t_pred = a * (n_arr ** b) / (p_arr ** c)
    ss_res = np.sum((t_arr - t_pred) ** 2)
    ss_tot = np.sum((t_arr - np.mean(t_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Качество аппроксимации (R²): {r_squared:.4f}")
    print()
    
    # Сохраняем формулу
    formula_data = {
        'formula': f'T(n, p) = {a:.6e} * n^{b:.4f} / p^{c:.4f}',
        'coefficients': {
            'a': float(a),
            'b': float(b),
            'c': float(c)
        },
        'r_squared': float(r_squared)
    }
    
    with open('formula.json', 'w') as f:
        json.dump(formula_data, f, indent=2)
    
    print("Формула сохранена в formula.json")
    
    return formula_data


def main():
    """
    Главная функция для запуска всех экспериментов
    """
    print("Численные эксперименты по параллельному умножению матриц")
    print(f"Доступно ядер CPU: {cpu_count()}")
    print()
    
    # Эксперимент 1: Зависимость от количества потоков
    results_threads = experiment_threads_dependency()
    
    # Эксперимент 2: Зависимость от размерности
    results_size = experiment_size_dependency()
    
    # Эксперимент 3: Линеаризация циклов
    results_linearization = experiment_linearization()
    
    # Вывод формулы
    formula = derive_formula(results_threads, results_size)
    
    print("\n" + "=" * 80)
    print("Все эксперименты завершены!")
    print("=" * 80)
    print()
    print("Созданные файлы:")
    print("  - results_threads.csv - зависимость от количества потоков")
    print("  - results_size.csv - зависимость от размерности")
    print("  - results_linearization.csv - сравнение линеаризаций")
    print("  - formula.json - формула для оценки времени выполнения")


if __name__ == '__main__':
    main()

