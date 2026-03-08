"""
Расширенные эксперименты для больших матриц (до 100000)
ВНИМАНИЕ: Может занять очень много времени!
"""

import numpy as np
import time
import csv
from matrix_multiply import (
    fill_matrix_divide_conquer,
    multiply_matrices_parallel,
    multiply_matrices_sequential,
    verify_result
)
from multiprocessing import cpu_count


def run_extended_size_experiment():
    """
    Расширенный эксперимент с размерами до 100000
    """
    print("=" * 80)
    print("Расширенный эксперимент: размеры матриц до 100000")
    print("=" * 80)
    print("ВНИМАНИЕ: Это может занять очень много времени!")
    print()
    
    # Размеры с логарифмическим шагом
    sizes = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    num_threads = cpu_count()
    
    results = []
    
    for size in sizes:
        print(f"\nРазмер матрицы: {size} x {size}")
        print("-" * 80)
        
        try:
            # Генерируем матрицы
            print("  Генерация матриц...")
            start_gen = time.perf_counter()
            A = fill_matrix_divide_conquer(size)
            B = fill_matrix_divide_conquer(size, seed=100)
            gen_time = time.perf_counter() - start_gen
            print(f"  Время генерации: {gen_time:.2f} с")
            
            # Последовательная версия (только для небольших размеров)
            if size <= 5000:
                print("  Последовательное умножение...")
                elapsed_seq, C_seq = measure_time(multiply_matrices_sequential, A, B)
                if not verify_result(A, B, C_seq):
                    print("  ОШИБКА: Неверный результат последовательной версии!")
                print(f"  Время: {elapsed_seq:.6f} с")
            else:
                elapsed_seq = None
                print("  Пропущено (слишком большой размер для последовательной версии)")
            
            # Параллельная версия
            print(f"  Параллельное умножение ({num_threads} потоков)...")
            elapsed_par, C_par = measure_time(
                multiply_matrices_parallel, A, B, num_threads
            )
            if not verify_result(A, B, C_par):
                print("  ОШИБКА: Неверный результат параллельной версии!")
            print(f"  Время: {elapsed_par:.6f} с")
            
            if elapsed_seq:
                speedup = elapsed_seq / elapsed_par
                print(f"  Ускорение: {speedup:.2f}x")
            else:
                speedup = None
            
            results.append({
                'size': size,
                'generation_time': gen_time,
                'sequential_time': elapsed_seq,
                'parallel_time': elapsed_par,
                'speedup': speedup,
                'threads': num_threads
            })
            
        except MemoryError:
            print(f"  ОШИБКА: Недостаточно памяти для матрицы {size}x{size}")
            break
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            break
    
    # Сохраняем результаты
    with open('results_extended_size.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'generation_time', 
                                                'sequential_time', 'parallel_time', 
                                                'speedup', 'threads'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nРезультаты сохранены в results_extended_size.csv")
    return results


def measure_time(func, *args, **kwargs):
    """Измерение времени выполнения функции"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start), result


if __name__ == '__main__':
    print("Расширенные эксперименты по умножению матриц")
    print(f"Доступно ядер CPU: {cpu_count()}")
    print()
    
    response = input("Вы уверены, что хотите запустить эксперименты до 100000? (yes/no): ")
    if response.lower() != 'yes':
        print("Эксперименты отменены.")
        exit(0)
    
    results = run_extended_size_experiment()
    
    print("\n" + "=" * 80)
    print("Эксперименты завершены!")
    print("=" * 80)

