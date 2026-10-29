# Local Estimates for Lighting

Research prototypes from a master's thesis in the Data Engineering programme at
National University of Science and Technology MISIS: **Mathematical Modeling of
Scene Brightness Distribution and Realistic Synthetic Image Rendering**.

The project explores Monte Carlo local estimates for illumination and the use of
denoising in synthetic image rendering. It includes numerical experiments,
visualizations, a Cornell box renderer, and the thesis itself.

## Repository contents

| Path | Contents |
| --- | --- |
| [`docs/lacal-estimates-thesis.pdf`](docs/lacal-estimates-thesis.pdf) | Original thesis |
| [`docs/local-estimates-thesis-eng.pdf`](docs/local-estimates-thesis-eng.pdf) | English thesis |
| [`docs/presentation.pptx`](docs/presentation.pptx) | Thesis presentation |
| [`pyton_code/sobolev_task`](pyton_code/sobolev_task) | Python local-estimate curve demonstration |
| [`pyton_code/double_local_estimate`](pyton_code/double_local_estimate) | Randomized 3D visualization of a double local estimate |
| [`pyton_code/denoising`](pyton_code/denoising) | Taichi renderer with Intel Open Image Denoise and Pygame display |
| [`cpp_code/sobolev_task`](cpp_code/sobolev_task) | C++ timing experiment for Sobolev and local-estimate formulas |
| [`cpp_code/denoising`](cpp_code/denoising) | C++ image denoising example using OIDN and FreeImage |
| [`matlab_code`](matlab_code) | MATLAB illumination comparison experiment |

The Python directory is named **`pyton_code`** in this repository. Each Python
example has its own `requirements.txt`; there is no root requirements file.

## Run a Python example

Use a separate virtual environment for each example. Python 3.10 or 3.11 is a
reasonable starting point for the historical dependency pins; these are not a
claim of compatibility with every current interpreter or GPU driver.

For the standalone illumination plot, run from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r pyton_code/sobolev_task/requirements.txt
python pyton_code/sobolev_task/main.py
```

On Windows, activate with `.venv\Scripts\activate` instead. The plotting examples
need a graphical desktop for interactive windows. For a headless smoke check,
use `MPLBACKEND=Agg python pyton_code/sobolev_task/main.py`.

Follow each example's README for its parameters and dependencies. The denoising
example additionally needs a working Taichi GPU backend and the native Intel
Open Image Denoise runtime expected by its Python wrapper.

## C++ and MATLAB

The C++ timing example uses CMake, a C++14 compiler, and Boost (including the
`system` component requested by the current CMake file):

```sh
cmake -S cpp_code/sobolev_task -B build/sobolev_task
cmake --build build/sobolev_task
./build/sobolev_task/LocalEstimatesLighting
```

The C++ denoiser is a separate source example without a supplied build target;
it requires Intel Open Image Denoise and FreeImage headers and libraries.
Run `matlab_code/illumination_comparison.m` in MATLAB for the MATLAB experiment.

## Numerical scope

These are thesis experiments, not a validated general-purpose renderer or a
single interchangeable implementation of the same equation. In particular:

- The Python `sobolev_task` script plots only its simplified local-estimate
  expression; the Sobolev integral is implemented in C++ and MATLAB.
- Its `1 / r` kernel is undefined at `r = 0`; the demo starts at a positive
  distance and rejects invalid inputs. Reflection weights are constant or
  specified per collision, independently of the number of rays.
- The Python, C++, and MATLAB experiments use different expressions and
  parameters. Their numerical results and timings require a method-specific
  review before being compared or used as physical predictions.
- The random visualization has no fixed seed, so successive runs differ.

## Checks

Run the Python local-estimate regression checks from the repository root. They
need only NumPy, using the same version as the plotting example:

```sh
python -m pip install numpy==2.0.0
python -m unittest discover -s tests -v
python -m compileall -q pyton_code/sobolev_task/main.py tests
```

The [Python checks workflow](.github/workflows/python-checks.yml) runs these
checks on Python 3.11 for pushes and pull requests. It does not install rendering
dependencies or validate the C++, MATLAB, GPU, or interactive plotting examples.

## Attribution

The Monte Carlo rendering code is based in part on
[HK-SHAO's Cornell box example](https://github.com/HK-SHAO/RayTracingPBR/blob/taichi-dev/examples/cornell_box/cornell_box_shortest.py)
from [RayTracingPBR](https://github.com/HK-SHAO/RayTracingPBR).
