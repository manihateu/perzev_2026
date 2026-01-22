# Лабораторная работа 2: Параллельная сортировка

## Описание

Реализация трех алгоритмов параллельной сортировки на C++ с использованием Threads:
1. Пузырьковая сортировка чет-нечет (Odd-Even Sort)
2. Быстрая сортировка (Quick Sort)
3. Поразрядная сортировка (Radix Sort)

## Требования

- Компилятор C++ с поддержкой C++17 (g++, clang++)
- Стандартная библиотека с поддержкой `<thread>`, `<mutex>`, `<condition_variable>`

## Сборка

```bash
make
```

или вручную:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread -c sorting_algorithms.cpp -o sorting_algorithms.o
g++ -std=c++17 -O2 -Wall -Wextra -pthread -c experiment.cpp -o experiment.o
g++ -std=c++17 -O2 -Wall -Wextra -pthread -o experiment sorting_algorithms.o experiment.o
```

## Запуск экспериментов

```bash
make run
```

или

```bash
./experiment
```

## Результаты

После выполнения экспериментов будут созданы файлы:
- `results_threads.csv` - результаты зависимости скорости от количества потоков
- `results_comparison.csv` - сравнение алгоритмов при равном количестве потоков

## Параметры экспериментов

- Размеры массивов: 100, 200, 500, 1000, 1010 элементов
- Количество потоков: 1, 2, 4, 8 (и более, в зависимости от системы)
- Каждый тест выполняется 5 раз для усреднения результатов

## Очистка

```bash
make clean
```

