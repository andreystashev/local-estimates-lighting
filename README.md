# local-estimates-lighting
Diploma project from the "Data Engineering" master's degree programme at the National University of Science and Technology MISIS.

## Описание
Выпускная квалификационная работа магистра на тему: "Математическое моделирование распределения яркости по сцене и реалистический вывод синтетического изображения".

В данной работе исследованы математические методы моделирования распределения яркости в трехмерных сценах, с особым акцентом на "локальные оценки метода Монте-Карло". Целью работы является повышение точности и эффективности визуализации синтетических изображений путем устранения зашумленности и ускорения процесса рендеринга.

## Структура репозитория
- **/docs**: текст проекта.
  - `thesis.pdf`
  - `presentation.pptx`
- **/cpp_code**: исходный код на C++ для замера скорости метода.
  - `main.cpp`
  - `CMakeLists.txt` 
- **/python_code**: исходный код на Python для денойзинга.
  - `main.py`
  - `requirements.txt`
  
  Часть кода, связанная с процессом рендеринга методом Монте-Карло использована [отсюда](https://github.com/HK-SHAO/RayTracingPBR/blob/taichi-dev/examples/cornell_box/cornell_box_shortest.py).
  Автор [HK-SHAO](https://github.com/HK-SHAO).
  
- **/matlab_code**: исходный код matlab для замера скорости метода.
  - `illumination_comparison.m`
  - `README.md`
- **README.md**: информация о проекте.

## Установка
### Зависимости C++
1. Установите компилятор C++ (например, GCC).
2. Установите CMake.

### Зависимости Python
1. Установите Python 3.9 или выше.
2. Установите зависимости:
   ```sh
   pip install -r requirements.txt
