#ifndef SORTING_ALGORITHMS_H
#define SORTING_ALGORITHMS_H

#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <climits>

/**
 * ѕараллельна€ пузырькова€ сортировка чет-нечет (Odd-Even Sort)
 *
 * јлгоритм работает в фазах:
 * - Ќечетна€ фаза: сравниваютс€ пары (1,2), (3,4), (5,6), ...
 * - „етна€ фаза: сравниваютс€ пары (0,1), (2,3), (4,5), ...
 *
 * @param arr - массив дл€ сортировки
 * @param num_threads - количество потоков
 */
void odd_even_sort_parallel(std::vector<int>& arr, int num_threads);

/**
 * ѕараллельна€ быстра€ сортировка (Quick Sort)
 *
 * »спользует рекурсивное разделение массива с параллельной обработкой подмассивов
 *
 * @param arr - массив дл€ сортировки
 * @param num_threads - количество потоков
 */
void quick_sort_parallel(std::vector<int>& arr, int num_threads);

/**
 * ѕараллельна€ поразр€дна€ сортировка (Radix Sort)
 *
 * —ортирует числа по разр€дам, параллелизу€ обработку разных частей массива
 *
 * @param arr - массив дл€ сортировки
 * @param num_threads - количество потоков
 */
void radix_sort_parallel(std::vector<int>& arr, int num_threads);

/**
 * ѕроверка корректности сортировки
 *
 * @param arr - отсортированный массив
 * @return true если массив отсортирован, false иначе
 */
bool is_sorted(const std::vector<int>& arr);

/**
 *  опирование массива
 */
std::vector<int> copy_array(const std::vector<int>& arr);

#endif // SORTING_ALGORITHMS_H

