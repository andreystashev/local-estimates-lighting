# C++ illumination timing experiment

`main.cpp` evaluates a Sobolev expression using Boost Bessel functions and a
separate simplified local-estimate expression, then prints their values and
execution times in microseconds.

From the repository root:

```sh
cmake -S cpp_code/sobolev_task -B build/sobolev_task
cmake --build build/sobolev_task
./build/sobolev_task/LocalEstimatesLighting
```

Requirements: CMake 3.10+, a C++14 compiler, and Boost, including the `system`
component requested by `CMakeLists.txt`. On multi-configuration generators, the
executable may be under a configuration directory such as `Debug` or `Release`.

Edit the source parameters to change the plane heights, reflection coefficients,
distance, ray count, or collision count. The Sobolev loop currently uses a step
of `0.01` up to `10000`; it can be slow. Its accumulation does not multiply by the
integration step as the MATLAB version does. Review the quadrature and align the
formulas and parameters before comparing numerical results or timing claims.
