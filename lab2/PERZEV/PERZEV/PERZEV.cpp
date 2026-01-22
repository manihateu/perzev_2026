#include "sorting_algorithms.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>

/**
 * Генерация случайного массива заданного размера
 */
std::vector<int> generate_random_array(int size, int seed = 42) {
    std::vector<int> arr(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(1, 10000);

    for (int i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }

    return arr;
}

/**
 * Измерение времени выполнения функции сортировки
 */
template<typename Func>
double measure_time(Func func, std::vector<int>& arr) {
    auto start = std::chrono::high_resolution_clock::now();
    func(arr);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Возвращаем время в миллисекундах
}

/**
 * Эксперимент: зависимость скорости от количества потоков
 */
void experiment_threads_dependency() {
    std::cout << "=" << std::setw(80) << std::setfill('=') << "\n";
    std::cout << "Эксперимент 1: Зависимость скорости от количества потоков\n";
    std::cout << "=" << std::setw(80) << std::setfill('=') << "\n\n";

    // Размеры массивов для тестирования
    std::vector<int> sizes = { 100, 200, 500, 1000, 1010 };

    // Количество потоков для тестирования
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 4;
    std::vector<int> thread_counts = { 1, 2, 4 };
    if (max_threads >= 8) thread_counts.push_back(8);
    if (max_threads > 8) thread_counts.push_back(max_threads);

    // Ограничиваем до разумных значений
    thread_counts.erase(std::remove_if(thread_counts.begin(), thread_counts.end(),
        [](int t) { return t > 16; }), thread_counts.end());

    std::ofstream csv_file("results_threads.csv");
    csv_file << "Algorithm,Size,Threads,Time(ms),Speedup\n";

    for (int size : sizes) {
        std::cout << "\nРазмер массива: " << size << " элементов\n";
        std::cout << std::string(80, '-') << "\n";

        // Генерируем тестовый массив
        auto test_array = generate_random_array(size);

        // Тестируем каждый алгоритм
        struct Algorithm {
            std::string name;
            void (*func)(std::vector<int>&, int);
        };

        std::vector<Algorithm> algorithms = {
            {"Odd-Even Sort", odd_even_sort_parallel},
            {"Quick Sort", quick_sort_parallel},
            {"Radix Sort", radix_sort_parallel}
        };

        for (const auto& alg : algorithms) {
            std::cout << "\n" << alg.name << ":\n";

            double time_1_thread = 0.0;

            for (int threads : thread_counts) {
                if (threads > size / 2 && alg.name == "Odd-Even Sort") {
                    continue; // Пропускаем для маленьких массивов
                }

                // Выполняем несколько прогонов для усреднения
                const int num_runs = 5;
                std::vector<double> times;

                for (int run = 0; run < num_runs; ++run) {
                    auto arr = copy_array(test_array);
                    double time = measure_time([&](std::vector<int>& a) {
                        alg.func(a, threads);
                        }, arr);

                    // Проверяем корректность
                    if (!is_sorted(arr)) {
                        std::cerr << "ОШИБКА: Массив не отсортирован!\n";
                    }

                    times.push_back(time);
                }

                // Усредняем результаты
                double avg_time = 0.0;
                for (double t : times) {
                    avg_time += t;
                }
                avg_time /= times.size();

                if (threads == 1) {
                    time_1_thread = avg_time;
                }

                double speedup = (time_1_thread > 0) ? time_1_thread / avg_time : 1.0;

                std::cout << "  " << std::setw(2) << threads << " потоков: "
                    << std::fixed << std::setprecision(3) << std::setw(10) << avg_time
                    << " мс, ускорение: " << std::setprecision(2) << speedup << "x\n";

                csv_file << alg.name << "," << size << "," << threads << ","
                    << avg_time << "," << speedup << "\n";
            }
        }
    }

    csv_file.close();
    std::cout << "\nРезультаты сохранены в results_threads.csv\n";
}

/**
 * Эксперимент: сравнение алгоритмов при равном количестве потоков
 */
void experiment_algorithms_comparison() {
    std::cout << "\n\n" << std::string(80, '=') << "\n";
    std::cout << "Эксперимент 2: Сравнение алгоритмов при равном количестве потоков\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Размеры массивов
    std::vector<int> sizes = { 100, 200, 500, 1000, 1010 };

    // Количество потоков
    int max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) max_threads = 4;
    std::vector<int> thread_counts = { 1, 2, 4 };
    if (max_threads >= 8) thread_counts.push_back(8);

    std::ofstream csv_file("results_comparison.csv");
    csv_file << "Size,Threads,Odd-Even(ms),Quick(ms),Radix(ms),Best\n";

    for (int size : sizes) {
        std::cout << "\nРазмер массива: " << size << " элементов\n";
        std::cout << std::string(80, '-') << "\n";

        auto test_array = generate_random_array(size);

        for (int threads : thread_counts) {
            if (threads > size / 2) {
                continue; // Пропускаем для маленьких массивов
            }

            std::cout << "\nКоличество потоков: " << threads << "\n";

            struct Result {
                std::string name;
                double time;
            };

            std::vector<Result> results;

            // Тестируем Odd-Even Sort
            {
                const int num_runs = 5;
                std::vector<double> times;
                for (int run = 0; run < num_runs; ++run) {
                    auto arr = copy_array(test_array);
                    double time = measure_time([&](std::vector<int>& a) {
                        odd_even_sort_parallel(a, threads);
                        }, arr);
                    if (is_sorted(arr)) {
                        times.push_back(time);
                    }
                }
                if (!times.empty()) {
                    double avg = 0.0;
                    for (double t : times) avg += t;
                    avg /= times.size();
                    results.push_back({ "Odd-Even", avg });
                }
            }

            // Тестируем Quick Sort
            {
                const int num_runs = 5;
                std::vector<double> times;
                for (int run = 0; run < num_runs; ++run) {
                    auto arr = copy_array(test_array);
                    double time = measure_time([&](std::vector<int>& a) {
                        quick_sort_parallel(a, threads);
                        }, arr);
                    if (is_sorted(arr)) {
                        times.push_back(time);
                    }
                }
                if (!times.empty()) {
                    double avg = 0.0;
                    for (double t : times) avg += t;
                    avg /= times.size();
                    results.push_back({ "Quick", avg });
                }
            }

            // Тестируем Radix Sort
            {
                const int num_runs = 5;
                std::vector<double> times;
                for (int run = 0; run < num_runs; ++run) {
                    auto arr = copy_array(test_array);
                    double time = measure_time([&](std::vector<int>& a) {
                        radix_sort_parallel(a, threads);
                        }, arr);
                    if (is_sorted(arr)) {
                        times.push_back(time);
                    }
                }
                if (!times.empty()) {
                    double avg = 0.0;
                    for (double t : times) avg += t;
                    avg /= times.size();
                    results.push_back({ "Radix", avg });
                }
            }

            // Выводим результаты
            std::sort(results.begin(), results.end(),
                [](const Result& a, const Result& b) { return a.time < b.time; });

            std::string best = results.empty() ? "N/A" : results[0].name;

            std::cout << "  Результаты:\n";
            for (const auto& r : results) {
                std::cout << "    " << std::setw(12) << std::left << r.name << ": "
                    << std::fixed << std::setprecision(3) << std::setw(10) << r.time
                    << " мс";
                if (r.name == best) {
                    std::cout << " [ЛУЧШИЙ]";
                }
                std::cout << "\n";
            }

            // Сохраняем в CSV
            csv_file << size << "," << threads;
            for (const auto& r : results) {
                csv_file << "," << r.time;
            }
            // Дополняем пустыми значениями если алгоритмов меньше 3
            for (size_t i = results.size(); i < 3; ++i) {
                csv_file << ",";
            }
            csv_file << "," << best << "\n";
        }
    }

    csv_file.close();
    std::cout << "\nРезультаты сохранены в results_comparison.csv\n";
}

int main() {
    setlocale(LC_ALL, "ru");
    std::cout << "Численные эксперименты по параллельной сортировке\n";
    std::cout << "Максимальный размер массива: 1010 элементов\n";
    std::cout << "Доступно потоков: " << std::thread::hardware_concurrency() << "\n\n";

    // Эксперимент 1: Зависимость от количества потоков
    experiment_threads_dependency();

    // Эксперимент 2: Сравнение алгоритмов
    experiment_algorithms_comparison();

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Все эксперименты завершены!\n";
    std::cout << std::string(80, '=') << "\n";

    return 0;
}

