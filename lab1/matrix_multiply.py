"""
Модуль для умножения матриц с использованием multiprocessing
и подхода "разделяй и властвуй" с переупорядочиванием матрицы.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def matrix_multiply_naive(A, B):
    """Наивное умножение матриц для проверки корректности."""
    return np.dot(A, B)


def matrix_multiply_linearized_ijk(A, B, C):
    """Умножение матриц с линеаризацией циклов в порядке i-j-k."""
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_multiply_linearized_ikj(A, B, C):
    """Умножение матриц с линеаризацией циклов в порядке i-k-j (оптимизированный)."""
    n = A.shape[0]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_multiply_linearized_jik(A, B, C):
    """Умножение матриц с линеаризацией циклов в порядке j-i-k."""
    n = A.shape[0]
    for j in range(n):
        for i in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_multiply_linearized_jki(A, B, C):
    """Умножение матриц с линеаризацией циклов в порядке j-k-i."""
    n = A.shape[0]
    for j in range(n):
        for k in range(n):
            for i in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_multiply_linearized_kij(A, B, C):
    """Умножение матриц с линеаризацией циклов в порядке k-i-j."""
    n = A.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_multiply_linearized_kji(A, B, C):
    """Умножение матриц с линеаризацией циклов в порядке k-j-i."""
    n = A.shape[0]
    for k in range(n):
        for j in range(n):
            for i in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def multiply_chunk(args):
    """Умножение части матриц для параллельного выполнения."""
    A, B, start_row, end_row = args
    n = A.shape[0]
    C_chunk = np.zeros((end_row - start_row, n))
    
    for i in range(start_row, end_row):
        for k in range(n):
            for j in range(n):
                C_chunk[i - start_row, j] += A[i, k] * B[k, j]
    
    return start_row, C_chunk


def matrix_multiply_parallel(A, B, num_threads=None, linearization='ikj'):
    """
    Умножение матриц с использованием multiprocessing и подхода разделяй и властвуй.
    
    Args:
        A: первая матрица
        B: вторая матрица (переупорядочивается для оптимизации доступа к памяти)
        num_threads: количество потоков (по умолчанию - количество ядер CPU)
        linearization: тип линеаризации циклов ('ijk', 'ikj', 'jik', 'jki', 'kij', 'kji')
    
    Returns:
        Результат умножения матриц
    """
    if num_threads is None:
        num_threads = cpu_count()
    
    n = A.shape[0]
    
    # Переупорядочивание матрицы B для оптимизации доступа к памяти
    # Транспонируем B для лучшей локальности данных
    B_T = B.T.copy()
    
    # Разделяем работу между потоками по строкам
    chunk_size = max(1, n // num_threads)
    chunks = []
    
    for i in range(0, n, chunk_size):
        end_row = min(i + chunk_size, n)
        chunks.append((A, B_T, i, end_row))
    
    # Параллельное выполнение
    with Pool(processes=num_threads) as pool:
        results = pool.map(multiply_chunk, chunks)
    
    # Собираем результаты
    C = np.zeros((n, n))
    for start_row, chunk in results:
        C[start_row:start_row + chunk.shape[0], :] = chunk
    
    return C


def multiply_chunk_linearized(args):
    """Умножение части матриц с различными вариантами линеаризации."""
    A, B_T, start_row, end_row, linearization = args
    n = A.shape[0]
    C_chunk = np.zeros((end_row - start_row, n))
    
    if linearization == 'ijk':
        for i in range(start_row, end_row):
            for j in range(n):
                for k in range(n):
                    C_chunk[i - start_row, j] += A[i, k] * B_T[j, k]
    elif linearization == 'ikj':
        for i in range(start_row, end_row):
            for k in range(n):
                for j in range(n):
                    C_chunk[i - start_row, j] += A[i, k] * B_T[j, k]
    elif linearization == 'jik':
        for j in range(n):
            for i in range(start_row, end_row):
                for k in range(n):
                    C_chunk[i - start_row, j] += A[i, k] * B_T[j, k]
    elif linearization == 'jki':
        for j in range(n):
            for k in range(n):
                for i in range(start_row, end_row):
                    C_chunk[i - start_row, j] += A[i, k] * B_T[j, k]
    elif linearization == 'kij':
        for k in range(n):
            for i in range(start_row, end_row):
                for j in range(n):
                    C_chunk[i - start_row, j] += A[i, k] * B_T[j, k]
    elif linearization == 'kji':
        for k in range(n):
            for j in range(n):
                for i in range(start_row, end_row):
                    C_chunk[i - start_row, j] += A[i, k] * B_T[j, k]
    
    return start_row, C_chunk


def matrix_multiply_parallel_linearized(A, B, num_threads=None, linearization='ikj'):
    """
    Умножение матриц с использованием multiprocessing и различными вариантами линеаризации.
    
    Args:
        A: первая матрица
        B: вторая матрица
        num_threads: количество потоков
        linearization: тип линеаризации ('ijk', 'ikj', 'jik', 'jki', 'kij', 'kji')
    
    Returns:
        Результат умножения матриц
    """
    if num_threads is None:
        num_threads = cpu_count()
    
    n = A.shape[0]
    
    # Переупорядочивание матрицы B (транспонирование для лучшей локальности)
    B_T = B.T.copy()
    
    # Разделяем работу между потоками
    chunk_size = max(1, n // num_threads)
    chunks = []
    
    for i in range(0, n, chunk_size):
        end_row = min(i + chunk_size, n)
        chunks.append((A, B_T, i, end_row, linearization))
    
    # Параллельное выполнение
    with Pool(processes=num_threads) as pool:
        results = pool.map(multiply_chunk_linearized, chunks)
    
    # Собираем результаты
    C = np.zeros((n, n))
    for start_row, chunk in results:
        C[start_row:start_row + chunk.shape[0], :] = chunk
    
    return C

