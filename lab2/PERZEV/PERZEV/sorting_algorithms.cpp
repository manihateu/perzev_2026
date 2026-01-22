#include "sorting_algorithms.h"
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <barrier>

// Вспомогательные функции для пузырьковой сортировки чет-нечет
namespace {
    void odd_even_phase(std::vector<int>& arr, int start, int end, bool is_odd_phase,
        std::barrier<>& sync_barrier) {
        int n = arr.size();

        if (is_odd_phase) {
            // Нечетная фаза: пары (1,2), (3,4), (5,6), ...
            for (int i = start; i < end; i += 2) {
                if (i + 1 < n && arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                }
            }
        }
        else {
            // Четная фаза: пары (0,1), (2,3), (4,5), ...
            for (int i = start; i < end; i += 2) {
                if (i + 1 < n && arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                }
            }
        }

        sync_barrier.arrive_and_wait();
    }
}

void odd_even_sort_parallel(std::vector<int>& arr, int num_threads) {
    int n = arr.size();
    if (n <= 1) return;

    // Для маленьких массивов используем последовательную версию
    if (n < 10) {
        bool swapped = true;
        while (swapped) {
            swapped = false;
            for (int i = 0; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    swapped = true;
                }
            }
            for (int i = 1; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    swapped = true;
                }
            }
        }
        return;
    }

    // Ограничиваем количество потоков
    num_threads = std::min(num_threads, n / 2);
    if (num_threads < 1) num_threads = 1;

    // Синхронизация через мьютекс и условную переменную
    std::mutex sync_mutex;
    std::condition_variable cv;
    int phase = 0; // 0 - четная, 1 - нечетная
    int threads_waiting = 0;
    std::atomic<bool> global_swapped{ true };
    bool done = false;

    // Функция для работы потока
    auto thread_work = [&](int thread_id) {
        // Распределяем элементы между потоками
        int elements_per_thread = (n + num_threads - 1) / num_threads;
        int start = thread_id * elements_per_thread;
        int end = std::min(start + elements_per_thread, n);

        int iterations = 0;

        while (iterations < n) {
            bool local_swapped = false;

            // Четная фаза: пары (0,1), (2,3), (4,5), ...
            {
                std::unique_lock<std::mutex> lock(sync_mutex);
                cv.wait(lock, [&]() { return phase == 0 || done; });
                if (done) return;
            }

            for (int i = start; i < end; ++i) {
                if (i % 2 == 0 && i + 1 < n && arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    local_swapped = true;
                }
            }

            if (local_swapped) global_swapped.store(true);

            {
                std::unique_lock<std::mutex> lock(sync_mutex);
                threads_waiting++;
                if (threads_waiting == num_threads) {
                    phase = 1;
                    threads_waiting = 0;
                    cv.notify_all();
                }
                else {
                    cv.wait(lock, [&]() { return phase == 1 || done; });
                }
            }

            // Нечетная фаза: пары (1,2), (3,4), (5,6), ...
            local_swapped = false;
            for (int i = start; i < end; ++i) {
                if (i % 2 == 1 && i + 1 < n && arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    local_swapped = true;
                }
            }

            if (local_swapped) global_swapped.store(true);

            {
                std::unique_lock<std::mutex> lock(sync_mutex);
                threads_waiting++;
                if (threads_waiting == num_threads) {
                    bool should_continue = global_swapped.load();
                    global_swapped.store(false);

                    if (!should_continue && iterations > 0) {
                        done = true;
                        cv.notify_all();
                        return;
                    }

                    phase = 0;
                    threads_waiting = 0;
                    iterations++;
                    cv.notify_all();
                }
                else {
                    cv.wait(lock, [&]() { return phase == 0 || done; });
                }
            }
        }
        };

    // Запускаем потоки
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(thread_work, i);
    }

    // Ждем завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
}

// Вспомогательные функции для быстрой сортировки
namespace {
    int partition(std::vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; ++j) {
            if (arr[j] <= pivot) {
                ++i;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }

    void quick_sort_sequential(std::vector<int>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quick_sort_sequential(arr, low, pi - 1);
            quick_sort_sequential(arr, pi + 1, high);
        }
    }

    struct SortTask {
        int low;
        int high;
    };
}

void quick_sort_parallel(std::vector<int>& arr, int num_threads) {
    int n = arr.size();
    if (n <= 1) return;

    // Для маленьких массивов используем последовательную версию
    if (n < 100 || num_threads <= 1) {
        quick_sort_sequential(arr, 0, n - 1);
        return;
    }

    std::mutex task_mutex;
    std::vector<SortTask> tasks;
    std::atomic<int> active_threads{ 0 };
    std::condition_variable cv;
    bool all_done = false;

    // Инициализируем первую задачу
    tasks.push_back({ 0, n - 1 });

    auto worker = [&]() {
        while (true) {
            SortTask task;
            bool has_task = false;

            {
                std::unique_lock<std::mutex> lock(task_mutex);

                // Ждем задачи или завершения
                cv.wait(lock, [&]() {
                    return !tasks.empty() || all_done;
                    });

                if (all_done && tasks.empty()) {
                    break;
                }

                if (!tasks.empty()) {
                    task = tasks.back();
                    tasks.pop_back();
                    has_task = true;
                    active_threads++;
                }
            }

            if (has_task) {
                if (task.low < task.high) {
                    int pi = partition(arr, task.low, task.high);

                    // Добавляем новые задачи
                    {
                        std::lock_guard<std::mutex> lock(task_mutex);
                        if (task.low < pi - 1) {
                            tasks.push_back({ task.low, pi - 1 });
                        }
                        if (pi + 1 < task.high) {
                            tasks.push_back({ pi + 1, task.high });
                        }
                    }
                    cv.notify_all();
                }

                {
                    std::lock_guard<std::mutex> lock(task_mutex);
                    active_threads--;
                    if (active_threads == 0 && tasks.empty()) {
                        all_done = true;
                    }
                    cv.notify_all();
                }
            }
        }
        };

    // Запускаем потоки
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Ждем завершения
    for (auto& t : threads) {
        t.join();
    }
}

// Вспомогательные функции для поразрядной сортировки
namespace {
    int get_max(const std::vector<int>& arr) {
        int max_val = arr[0];
        for (int val : arr) {
            if (val > max_val) {
                max_val = val;
            }
        }
        return max_val;
    }

    void count_sort_radix(std::vector<int>& arr, int exp, int start, int end) {
        std::vector<int> output(end - start);
        std::vector<int> count(10, 0);

        // Подсчет количества элементов
        for (int i = start; i < end; ++i) {
            count[(arr[i] / exp) % 10]++;
        }

        // Изменяем count[i] так, чтобы он содержал позицию
        for (int i = 1; i < 10; ++i) {
            count[i] += count[i - 1];
        }

        // Строим выходной массив
        for (int i = end - 1; i >= start; --i) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }

        // Копируем обратно
        for (int i = start; i < end; ++i) {
            arr[i] = output[i - start];
        }
    }
}

void radix_sort_parallel(std::vector<int>& arr, int num_threads) {
    int n = arr.size();
    if (n <= 1) return;

    // Находим максимальное число для определения количества разрядов
    int max_val = get_max(arr);

    // Для каждого разряда выполняем сортировку подсчетом
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        // Параллелизуем подсчет по частям массива
        std::vector<std::vector<int>> local_counts(num_threads, std::vector<int>(10, 0));

        int elements_per_thread = (n + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;

        // Фаза 1: Подсчет в каждом потоке
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                int start = t * elements_per_thread;
                int end = std::min(start + elements_per_thread, n);

                for (int i = start; i < end; ++i) {
                    local_counts[t][(arr[i] / exp) % 10]++;
                }
                });
        }

        for (auto& t : threads) {
            t.join();
        }

        // Объединяем подсчеты
        std::vector<int> count(10, 0);
        for (int i = 0; i < num_threads; ++i) {
            for (int j = 0; j < 10; ++j) {
                count[j] += local_counts[i][j];
            }
        }

        // Преобразуем count в позиции
        for (int i = 1; i < 10; ++i) {
            count[i] += count[i - 1];
        }

        // Фаза 2: Перераспределение элементов
        // Для стабильности Radix Sort нужно обрабатывать элементы строго в обратном порядке
        // Поэтому эта фаза выполняется последовательно
        std::vector<int> output(n);
        std::vector<int> count_positions = count;

        // Обрабатываем массив в обратном порядке для сохранения стабильности
        for (int i = n - 1; i >= 0; --i) {
            int digit = (arr[i] / exp) % 10;
            output[count_positions[digit] - 1] = arr[i];
            count_positions[digit]--;
        }

        // Копируем обратно
        arr = output;
    }
}

bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

std::vector<int> copy_array(const std::vector<int>& arr) {
    return std::vector<int>(arr);
}

