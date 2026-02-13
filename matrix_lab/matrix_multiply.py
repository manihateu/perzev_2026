"""
Модуль для параллельного умножения матриц
Вариант: Разделяй и властвуй (заполнение), Multiprocessing, Переупорядочивание матрицы
"""

import numpy as np
from multiprocessing import Process, Queue, cpu_count
from typing import Tuple, List
import time


def fill_matrix_divide_conquer(size: int, seed: int = 42) -> np.ndarray:
    """
    Заполнение матрицы методом "Разделяй и властвуй"
    
    Алгоритм:
    1. Разделяем матрицу на 4 подматрицы
    2. Рекурсивно заполняем каждую подматрицу
    3. Объединяем результаты
    
    Args:
        size: размер матрицы (size x size)
        seed: начальное значение для генератора случайных чисел
        
    Returns:
        Заполненная матрица размером size x size
    """
    if size <= 1:
        return np.array([[np.random.RandomState(seed).randint(1, 100)]])
    
    if size <= 4:
        # Базовый случай: заполняем напрямую
        rng = np.random.RandomState(seed)
        return rng.randint(1, 100, size=(size, size))
    
    # Разделяем матрицу на 4 части
    half = size // 2
    remaining = size - half
    
    # Создаем 4 блока нужных размеров:
    # top_left: half x half
    # top_right: half x remaining
    # bottom_left: remaining x half
    # bottom_right: remaining x remaining
    
    # Рекурсивно заполняем квадратные блоки
    top_left = fill_matrix_divide_conquer(half, seed)
    bottom_right = fill_matrix_divide_conquer(remaining, seed + 3)
    
    # Для прямоугольных блоков используем рекурсию или прямое заполнение
    rng = np.random.RandomState(seed)
    
    # top_right: half x remaining
    if half > 4:
        top_right_temp = fill_matrix_divide_conquer(half, seed + 1)
        top_right = top_right_temp[:, :remaining]
    else:
        rng_tr = np.random.RandomState(seed + 1)
        top_right = rng_tr.randint(1, 100, size=(half, remaining))
    
    # bottom_left: remaining x half
    if remaining > 4:
        bottom_left_temp = fill_matrix_divide_conquer(remaining, seed + 2)
        bottom_left = bottom_left_temp[:remaining, :half]
    else:
        rng_bl = np.random.RandomState(seed + 2)
        bottom_left = rng_bl.randint(1, 100, size=(remaining, half))
    
    # Объединяем части
    top = np.hstack([top_left, top_right])
    bottom = np.hstack([bottom_left, bottom_right])
    result = np.vstack([top, bottom])
    
    return result


def reorder_matrix_for_cache(A: np.ndarray) -> np.ndarray:
    """
    Переупорядочивание матрицы для оптимизации кэша
    
    Транспонирует матрицу для улучшения доступа по столбцам
    (кэш-дружественный доступ).
    
    Args:
        A: исходная матрица
        
    Returns:
        Переупорядоченная (транспонированная) матрица
    """
    return A.T.copy()


def multiply_worker(start_row: int, end_row: int, A_data: List, 
                   B_data: List, result_queue: Queue):
    """
    Рабочая функция для умножения части матриц в отдельном процессе
    
    Args:
        start_row: начальная строка
        end_row: конечная строка (не включительно)
        A_data: данные матрицы A в виде списка
        B_data: данные матрицы B в виде списка
        result_queue: очередь для передачи результатов
    """
    # Восстанавливаем матрицы из списков
    A = np.array(A_data)
    B = np.array(B_data)
    
    n = A.shape[1]
    m = B.shape[1]
    
    # Вычисляем свою часть результата
    result_part = np.zeros((end_row - start_row, m))
    
    # Переупорядочиваем матрицу B для оптимизации доступа (транспонируем)
    B_T = B.T  # Транспонированная матрица для лучшего доступа по столбцам
    
    # Умножение с переупорядочиванием
    # Используем транспонированную B для доступа B[j, k] вместо B[k, j]
    for i in range(start_row, end_row):
        for k in range(n):
            a_ik = A[i, k]
            # Доступ к транспонированной B: B_T[j, k] = B[k, j]
            for j in range(m):
                result_part[i - start_row, j] += a_ik * B_T[j, k]
    
    result_queue.put((start_row, result_part.tolist()))


def multiply_matrices_parallel(A: np.ndarray, B: np.ndarray, 
                               num_threads: int = None) -> np.ndarray:
    """
    Параллельное умножение матриц с переупорядочиванием
    
    Использует multiprocessing для распределения работы между процессами.
    Каждый процесс обрабатывает свой диапазон строк результирующей матрицы.
    
    Args:
        A: первая матрица (n x m)
        B: вторая матрица (m x k)
        num_threads: количество процессов (по умолчанию - количество ядер CPU)
        
    Returns:
        Результирующая матрица (n x k)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Несовместимые размеры матриц для умножения")
    
    n, m = A.shape
    k = B.shape[1]
    
    if num_threads is None:
        num_threads = cpu_count()
    
    num_threads = min(num_threads, n)  # Не больше, чем строк
    
    if num_threads == 1:
        # Последовательная версия
        return multiply_matrices_sequential(A, B)
    
    # Распределяем строки между процессами
    rows_per_process = n // num_threads
    processes = []
    result_queue = Queue()
    
    # Конвертируем матрицы в списки для передачи между процессами
    A_list = A.tolist()
    B_list = B.tolist()
    
    # Запускаем процессы
    for i in range(num_threads):
        start_row = i * rows_per_process
        end_row = (i + 1) * rows_per_process if i < num_threads - 1 else n
        
        p = Process(target=multiply_worker, 
                   args=(start_row, end_row, A_list, B_list, result_queue))
        p.start()
        processes.append(p)
    
    # Собираем результаты
    result = np.zeros((n, k))
    for _ in range(num_threads):
        start_row, result_part = result_queue.get()
        result_part = np.array(result_part)
        end_row = start_row + result_part.shape[0]
        result[start_row:end_row, :] = result_part
    
    # Ждем завершения всех процессов
    for p in processes:
        p.join()
    
    return result


def multiply_matrices_sequential(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Последовательное умножение матриц с переупорядочиванием
    
    Args:
        A: первая матрица (n x m)
        B: вторая матрица (m x k)
        
    Returns:
        Результирующая матрица (n x k)
    """
    n, m = A.shape
    k = B.shape[1]
    result = np.zeros((n, k))
    
    # Переупорядочиваем матрицу B (транспонируем)
    B_T = B.T  # Транспонированная матрица
    
    # Классическое умножение с переупорядочиванием
    for i in range(n):
        for k_idx in range(m):
            a_ik = A[i, k_idx]
            # Доступ к транспонированной B: B_T[j, k_idx] = B[k_idx, j]
            for j in range(k):
                result[i, j] += a_ik * B_T[j, k_idx]
    
    return result


def multiply_matrices_linearized_v1(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Умножение матриц с линеаризацией циклов (вариант 1)
    
    Линеаризация: i, k, j -> общий индекс
    """
    n, m = A.shape
    k = B.shape[1]
    result = np.zeros((n, k))
    
    B_T = B.T  # Транспонированная матрица
    total = n * m * k
    
    for idx in range(total):
        i = idx // (m * k)
        remainder = idx % (m * k)
        k_idx = remainder // k
        j = remainder % k
        
        if i < n and k_idx < m and j < k:
            result[i, j] += A[i, k_idx] * B_T[j, k_idx]
    
    return result


def multiply_matrices_linearized_v2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Умножение матриц с линеаризацией циклов (вариант 2)
    
    Линеаризация: k, i, j -> общий индекс
    """
    n, m = A.shape
    k = B.shape[1]
    result = np.zeros((n, k))
    
    B_T = B.T  # Транспонированная матрица
    total = m * n * k
    
    for idx in range(total):
        k_idx = idx // (n * k)
        remainder = idx % (n * k)
        i = remainder // k
        j = remainder % k
        
        if k_idx < m and i < n and j < k:
            result[i, j] += A[i, k_idx] * B_T[j, k_idx]
    
    return result


def multiply_matrices_linearized_v3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Умножение матриц с линеаризацией циклов (вариант 3)
    
    Линеаризация: j, i, k -> общий индекс
    """
    n, m = A.shape
    k = B.shape[1]
    result = np.zeros((n, k))
    
    B_T = B.T  # Транспонированная матрица
    total = k * n * m
    
    for idx in range(total):
        j = idx // (n * m)
        remainder = idx % (n * m)
        i = remainder // m
        k_idx = remainder % m
        
        if j < k and i < n and k_idx < m:
            result[i, j] += A[i, k_idx] * B_T[j, k_idx]
    
    return result


def verify_result(A: np.ndarray, B: np.ndarray, result: np.ndarray, 
                  tolerance: float = 1e-5) -> bool:
    """
    Проверка корректности умножения матриц
    
    Args:
        A: первая матрица
        B: вторая матрица
        result: результат умножения
        tolerance: допустимая погрешность
        
    Returns:
        True если результат корректен, False иначе
    """
    expected = np.dot(A, B)
    return np.allclose(result, expected, atol=tolerance)

