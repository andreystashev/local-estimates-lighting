# Local-estimate illumination plot

`main.py` plots the simplified local-estimate expression used by this thesis
experiment. Despite the directory name, this Python script does **not** implement
the Sobolev integral; see the C++ and MATLAB experiments for that calculation.

From the repository root, install this example's dependencies in a virtual
environment and run:

```sh
python -m pip install -r pyton_code/sobolev_task/requirements.txt
python pyton_code/sobolev_task/main.py
```

Edit `Phi`, `N`, `T`, `rho`, and `r_values` in `main()` to change the flux, ray
count, collision count, reflection weight, and distances. The current prototype
sums the same deterministic term for each ray; increasing `N` changes loop work,
not the modeled random samples. Its values are not a convergence benchmark for
a complete Monte Carlo transport algorithm.

`calculate_illuminance(Phi, N, T, rho, r)` can also be imported without opening a
plot. `N` must be a positive integer, `T` a non-negative integer, `Phi` finite and
non-negative, and `r` finite and positive. The `1 / r` kernel is singular at zero.
`rho` accepts either a scalar weight or a sequence containing at least `T + 1`
weights, all between zero and one. The formula preserves the original weights;
it does not introduce cumulative reflection attenuation.
