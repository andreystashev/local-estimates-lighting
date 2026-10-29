# MATLAB illumination comparison

`illumination_comparison.m` evaluates a local-estimate expression and a Sobolev
expression, prints their execution times and timing ratio, and plots both curves.
The Sobolev expression uses MATLAB's `besselk` and `besselj` functions.

Open the script in MATLAB and run it. MATLAB R2016b or later is needed for local
functions in scripts. No external MATLAB toolbox is explicitly imported.

Edit `Phi`, `N`, `T`, `r`, `rho`, `h1`, `h2`, `rho1`, and `rho2` to change the
experiment. The local kernel contains `1 / r`, so distances must stay positive.
In this historical script, `T` must not exceed `N` because reflection weights are
allocated using `N + 1` entries.

The implementations differ from the Python and C++ experiments, including the
reflection weights and integration range. Review their formulas and numerical
integration settings before treating the timing ratio as a comparison of
methods with equivalent accuracy.
