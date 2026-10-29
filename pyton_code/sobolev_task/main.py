import numpy as np


def calculate_illuminance(Phi, N, T, rho, r):
    """Evaluate the prototype local-estimate formula at a positive distance.

    ``rho`` is a constant reflection weight or a one-dimensional sequence with
    a weight for each of the ``T + 1`` collisions used by the original formula.
    """
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not isinstance(T, (int, np.integer)) or T < 0:
        raise ValueError("T must be a non-negative integer")
    if not np.isfinite(Phi) or Phi < 0:
        raise ValueError("Phi must be finite and non-negative")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r must be finite and positive")

    weights = np.asarray(rho, dtype=float)
    if weights.ndim == 0:
        weights = np.full(T + 1, weights.item())
    elif weights.ndim != 1 or len(weights) < T + 1:
        raise ValueError("rho must be a scalar or contain at least T + 1 weights")
    if not np.all(np.isfinite(weights)) or np.any((weights < 0) | (weights > 1)):
        raise ValueError("rho weights must be finite and between 0 and 1")

    result = 0
    pi_val = np.pi
    # Keep the original ray/collision loops for this timing demonstration.
    for n in range(1, N + 1):
        for i in range(T + 1):
            F = 1 / (pi_val * r)
            result += Phi / (pi_val * N) * weights[i] * F

    illuminance = result
    return illuminance


def main():
    import matplotlib.pyplot as plt

    Phi = 1.0  # Source luminous flux.
    N = 20  # Number of rays.
    T = 5  # Number of collisions.
    rho = 0.5  # Constant reflection weight.

    # The prototype's 1 / r kernel is singular at zero.
    r_values = np.linspace(0.1, 10, 100)
    illuminance_values = [
        calculate_illuminance(Phi, N, T, rho, r) for r in r_values
    ]

    plt.plot(r_values, illuminance_values, linewidth=2)
    plt.xlabel("Distance r")
    plt.ylabel("Illuminance E(r)")
    plt.title("Prototype illuminance distribution")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

