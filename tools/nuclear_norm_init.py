import numpy as np
from typing import Tuple


def nuclear_norm_init(
    X: np.ndarray, m: int, n: int, r: int, seed: int = 1293871, verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    Theta1 = rng.standard_normal(size=(m, n))
    Theta2, _ = nmd_nuclear_bt(X, Theta1, 3, verbose=verbose)
    ua, sa, va = np.linalg.svd(Theta2, full_matrices=False)
    sa = np.diag(sa)[:r, :r]
    W0 = ua[:, :r]
    H0 = sa @ va[:r, :]
    return W0, H0


def nmd_nuclear_bt(
    X: np.typing.ArrayLike,
    Theta: np.typing.ArrayLike,
    max_iter: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, list[float]]:
    assert np.all(X >= 0)

    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    alpha = 1 / 1**0.1  # Initial choice for alpha
    Theta[x_is_pos] = X[x_is_pos]  # Set the fixed components of Theta
    Theta[x_is_zero] = np.minimum(0, Theta[x_is_zero])

    nuclear_norms = []

    for i in range(max_iter):
        if verbose:
            print(f"Iteration { i + 1 } out of { max_iter }")
        U, D, Vt = np.linalg.svd(Theta, full_matrices=False)
        nuclear_norms.append(np.sum(np.diag(D)))  # Nuclear norm eval

        # backtracking
        if i > 0 and nuclear_norms[i] < nuclear_norms[i - 1]:
            alpha *= 1.2
        else:
            alpha *= 0.7

        # Update Theta
        # Theta = Theta - alpha * (U @ Vt)
        Theta -= alpha * (U @ Vt)

        # Project Theta
        Theta[x_is_pos] = X[x_is_pos]
        Theta[x_is_zero] = np.minimum(0, Theta[x_is_zero])

    return Theta, nuclear_norms
