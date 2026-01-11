"""
Скрипт для проведения численных экспериментов по умножению матриц.
"""

import numpy as np
import time
import json
import csv
from matrix_multiply import (
    matrix_multiply_naive,
    matrix_multiply_parallel_linearized
)
from multiprocessing import cpu_count


def generate_matrices(n):
    """Генерация двух случайных матриц размерности n x n."""
    np.random.seed(42)  # Для воспроизводимости результатов
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    return A, B


def measure_time(func, *args, **kwargs):
    """Измерение времени выполнения функции."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def experiment_varying_threads():
    """Эксперимент: зависимость времени от количества потоков."""
    print("=" * 60)
    print("Эксперимент 1: Зависимость от количества потоков")
    print("=" * 60)
    
    # Размеры матриц для тестирования
    sizes = [10, 50, 100, 200, 500, 1000]
    max_threads = min(cpu_count(), 8)  # Ограничиваем до 8 потоков
    thread_counts = [1, 2, 4, max_threads]
    
    results = []
    
    for size in sizes:
        print(f"\nРазмер матрицы: {size}x{size}")
        A, B = generate_matrices(size)
        
        # Проверка корректности с помощью numpy
        reference = matrix_multiply_naive(A, B)
        
        for num_threads in thread_counts:
            if num_threads > size:
                continue
            
            times = []
            for _ in range(3):  # 3 прогона для усреднения
                elapsed, result = measure_time(
                    matrix_multiply_parallel_linearized,
                    A, B, num_threads=num_threads, linearization='ikj'
                )
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Проверка корректности
            error = np.abs(result - reference).max()
            
            result_entry = {
                'size': size,
                'threads': num_threads,
                'time': avg_time,
                'std': std_time,
                'error': error
            }
            results.append(result_entry)
            
            print(f"  Потоков: {num_threads:2d}, Время: {avg_time:.6f}±{std_time:.6f} сек, Ошибка: {error:.2e}")
    
    return results


def experiment_varying_sizes():
    """Эксперимент: зависимость времени от размера матрицы."""
    print("\n" + "=" * 60)
    print("Эксперимент 2: Зависимость от размера матрицы")
    print("=" * 60)
    
    # Используем логарифмическую шкалу для размеров от 10 до 100000
    # Но ограничиваем максимальный размер в зависимости от доступной памяти
    sizes = []
    
    # Малые размеры
    sizes.extend([10, 20, 50, 100])
    
    # Средние размеры
    sizes.extend([200, 500, 1000, 2000])
    
    # Большие размеры (если позволяет память)
    try:
        # Проверяем доступную память, пробуя создать тестовую матрицу
        test_size = 5000
        test_matrix = np.random.rand(test_size, test_size).astype(np.float32)
        del test_matrix
        sizes.extend([5000, 10000])
    except MemoryError:
        print("  Предупреждение: ограниченная память, пропускаем большие размеры")
    
    try:
        test_size = 20000
        test_matrix = np.random.rand(test_size, test_size).astype(np.float32)
        del test_matrix
        sizes.extend([20000, 50000])
    except MemoryError:
        pass
    
    try:
        test_size = 100000
        test_matrix = np.random.rand(test_size, test_size).astype(np.float32)
        del test_matrix
        sizes.append(100000)
    except MemoryError:
        pass
    
    sizes = sorted(set(sizes))
    num_threads = min(cpu_count(), 4)
    results = []
    
    for size in sizes:
        print(f"\nРазмер матрицы: {size}x{size}")
        try:
            A, B = generate_matrices(size)
            
            # Для больших матриц делаем меньше прогонов
            num_runs = 3 if size <= 1000 else 1
            
            times = []
            for _ in range(num_runs):
                elapsed, _ = measure_time(
                    matrix_multiply_parallel_linearized,
                    A, B, num_threads=num_threads, linearization='ikj'
                )
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times) if len(times) > 1 else 0.0
            
            result_entry = {
                'size': size,
                'threads': num_threads,
                'time': avg_time,
                'std': std_time
            }
            results.append(result_entry)
            
            print(f"  Время: {avg_time:.6f}±{std_time:.6f} сек")
            
            # Освобождаем память
            del A, B
        except MemoryError:
            print(f"  Пропущено из-за нехватки памяти")
            break
        except Exception as e:
            print(f"  Ошибка: {e}")
            break
    
    return results


def experiment_linearization():
    """Эксперимент: сравнение различных вариантов линеаризации циклов."""
    print("\n" + "=" * 60)
    print("Эксперимент 3: Сравнение вариантов линеаризации циклов")
    print("=" * 60)
    
    sizes = [10, 50, 100, 200, 500, 1000]
    linearizations = ['ijk', 'ikj', 'jik', 'jki', 'kij', 'kji']
    num_threads = min(cpu_count(), 4)
    
    results = []
    
    for size in sizes:
        print(f"\nРазмер матрицы: {size}x{size}")
        A, B = generate_matrices(size)
        reference = matrix_multiply_naive(A, B)
        
        best_time = float('inf')
        best_linearization = None
        
        for lin in linearizations:
            times = []
            for _ in range(3):
                elapsed, result = measure_time(
                    matrix_multiply_parallel_linearized,
                    A, B, num_threads=num_threads, linearization=lin
                )
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            error = np.abs(result - reference).max()
            
            if avg_time < best_time:
                best_time = avg_time
                best_linearization = lin
            
            result_entry = {
                'size': size,
                'linearization': lin,
                'time': avg_time,
                'std': std_time,
                'error': error
            }
            results.append(result_entry)
            
            print(f"  {lin}: {avg_time:.6f}±{std_time:.6f} сек, Ошибка: {error:.2e}")
        
        print(f"  Лучший вариант: {best_linearization} ({best_time:.6f} сек)")
    
    return results


def fit_formula(results_threads, results_sizes):
    """Подбор формулы для оценки времени выполнения."""
    print("\n" + "=" * 60)
    print("Анализ результатов и вывод формулы")
    print("=" * 60)
    
    # Анализ зависимости от размера
    sizes = np.array([r['size'] for r in results_sizes])
    times = np.array([r['time'] for r in results_sizes])
    
    # Логарифмическая регрессия: time = a * size^b
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    
    # Линейная регрессия в логарифмическом масштабе
    coeffs = np.polyfit(log_sizes, log_times, 1)
    b = coeffs[0]  # Степень
    log_a = coeffs[1]
    a = np.exp(log_a)
    
    # Вычисляем R² для оценки качества аппроксимации
    predicted_log_times = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_times - predicted_log_times) ** 2)
    ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nЗависимость от размера матрицы:")
    print(f"  Время ≈ {a:.2e} * N^{b:.2f}")
    print(f"  (где N - размерность матрицы)")
    print(f"  Качество аппроксимации (R²): {r_squared:.4f}")
    
    # Анализ зависимости от количества потоков
    thread_data = {}
    for r in results_threads:
        size = r['size']
        if size not in thread_data:
            thread_data[size] = []
        thread_data[size].append((r['threads'], r['time']))
    
    print(f"\nЗависимость от количества потоков:")
    efficiency_data = []
    
    for size in sorted(thread_data.keys())[-3:]:  # Показываем для 3 наибольших размеров
        data = sorted(thread_data[size])
        threads = [t for t, _ in data]
        times = [t for _, t in data]
        
        # Ускорение относительно 1 потока
        base_time = times[0]
        speedups = [base_time / t for t in times]
        efficiencies = [sp / th for sp, th in zip(speedups, threads)]
        
        print(f"  Размер {size}x{size}:")
        for th, sp, eff in zip(threads, speedups, efficiencies):
            print(f"    {th} потоков: ускорение {sp:.2f}x, эффективность {eff:.2f}")
            efficiency_data.append((size, th, sp, eff))
    
    # Подбор функции ускорения от количества потоков
    if efficiency_data:
        # Анализируем эффективность для разных размеров
        avg_efficiency = {}
        for size, th, sp, eff in efficiency_data:
            if th not in avg_efficiency:
                avg_efficiency[th] = []
            avg_efficiency[th].append(eff)
        
        print(f"\nСредняя эффективность использования потоков:")
        for th in sorted(avg_efficiency.keys()):
            avg_eff = np.mean(avg_efficiency[th])
            print(f"  {th} потоков: {avg_eff:.2f}")
        
        # Подбор формулы для функции ускорения
        # Обычно ускорение имеет вид: speedup = P^α, где α < 1
        thread_counts = sorted(avg_efficiency.keys())
        if len(thread_counts) > 1:
            # Используем средние ускорения для подбора
            avg_speedups = {}
            for size, th, sp, eff in efficiency_data:
                if th not in avg_speedups:
                    avg_speedups[th] = []
                avg_speedups[th].append(sp)
            
            if len(avg_speedups) > 1:
                log_threads = np.log(thread_counts[1:])  # Пропускаем 1 поток
                log_speedups = [np.log(np.mean(avg_speedups[th])) for th in thread_counts[1:]]
                
                if len(log_threads) > 0:
                    alpha_coeffs = np.polyfit(log_threads, log_speedups, 1)
                    alpha = alpha_coeffs[0]
                    
                    print(f"\nФункция ускорения от количества потоков:")
                    print(f"  speedup(P) ≈ P^{alpha:.3f}")
    
    # Общая формула
    print(f"\n" + "=" * 60)
    print(f"ОБЩАЯ ФОРМУЛА ОЦЕНКИ ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
    print(f"=" * 60)
    print(f"  T(N, P) ≈ {a:.2e} * N^{b:.2f} / P^α")
    print(f"  где:")
    print(f"    N - размерность матрицы")
    print(f"    P - количество потоков")
    print(f"    α - коэффициент эффективности параллелизации (обычно 0.6-0.9)")
    print(f"")
    print(f"  Альтернативная форма:")
    print(f"    T(N, P) ≈ {a:.2e} * N^{b:.2f} * P^(-α)")
    print(f"=" * 60)
    
    return {
        'a': float(a),
        'b': float(b),
        'r_squared': float(r_squared),
        'formula_size': f"{a:.2e} * N^{b:.2f}",
        'formula_full': f"{a:.2e} * N^{b:.2f} / P^α"
    }


def save_results(results_threads, results_sizes, results_linearization, formula):
    """Сохранение результатов в файлы."""
    # Сохранение в JSON
    with open('lab1/results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'threads_experiment': results_threads,
            'sizes_experiment': results_sizes,
            'linearization_experiment': results_linearization,
            'formula': formula
        }, f, indent=2, ensure_ascii=False)
    
    # Сохранение в CSV для удобного анализа
    with open('lab1/results_threads.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'threads', 'time', 'std', 'error'])
        writer.writeheader()
        writer.writerows(results_threads)
    
    with open('lab1/results_sizes.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'threads', 'time', 'std'])
        writer.writeheader()
        writer.writerows(results_sizes)
    
    with open('lab1/results_linearization.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'linearization', 'time', 'std', 'error'])
        writer.writeheader()
        writer.writerows(results_linearization)
    
    print("\nРезультаты сохранены в:")
    print("  - lab1/results.json")
    print("  - lab1/results_threads.csv")
    print("  - lab1/results_sizes.csv")
    print("  - lab1/results_linearization.csv")


def main():
    """Основная функция для запуска всех экспериментов."""
    print("Начало численных экспериментов по умножению матриц")
    print(f"Количество доступных ядер CPU: {cpu_count()}")
    
    # Эксперимент 1: Зависимость от количества потоков
    results_threads = experiment_varying_threads()
    
    # Эксперимент 2: Зависимость от размера матрицы
    results_sizes = experiment_varying_sizes()
    
    # Эксперимент 3: Сравнение линеаризаций
    results_linearization = experiment_linearization()
    
    # Анализ и вывод формулы
    formula = fit_formula(results_threads, results_sizes)
    
    # Сохранение результатов
    save_results(results_threads, results_sizes, results_linearization, formula)
    
    print("\n" + "=" * 60)
    print("Эксперименты завершены!")
    print("=" * 60)


if __name__ == '__main__':
    main()

