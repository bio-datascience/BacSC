import numpy as np
import numpy.typing as npt
from time import perf_counter
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD


def nmd_t(
    X: npt.NDArray[np.float_],
    r: int,
    W0: npt.NDArray[np.float_] = None,
    H0: npt.NDArray[np.float_] = None,
    beta1: float = 0.95,
    regularization_lambda: float = 0.0001,
    max_iters: int = 1000,
    tol: float = 1e-4,
    tol_over_10iters: float = 1e-5,
    verbose: bool = True,
) -> (npt.NDArray[np.float_], list[float], int, list[float]):
    """NMD using three-block alternating minimization with Tikhonov regularization

    Adaption of 3B-NMD, adding Tikhonov regularization on W and H as well as
    momentum steps on W and H with a small negative momentum parameter.

    Args:
        X (npt.NDArray[np.float_]): (m, n) sparse non-negative matrix
        r (int): approximation rank
        W0 (npt.NDArray[np.float_], optional): initial W.
            Defaults to np.random.randn(m, r) if none is provided.
        H0 (npt.NDArray[np.float_], optional): initial H.
            Defaults to np.random.randn(r, n) if none is provided.
        beta1 (float, optional): momentum parameter. Defaults to 0.95.
        regularization_lambda (float, optional): regularization parameter. Defaults to 0.0001.
        max_iters (int, optional): maximum number of iterations. Defaults to 1000.
        tol (float, optional):  stopping criterion on the relative error:
            ||X-max(0,WH)||/||X|| < tol. Defaults to 1e-4.
        tol_over_10iters (float, optional): stopping criterion tolerance on 10
            successive errors: abs(errors[i] - errors[i-10]) < tol_err.
            Defaults to 1e-5.
        verbose (bool, optional): print information to console. Defaults to True.

    Returns:
        (npt.NDArray[np.float_], list[float], int, list[float]):
            Theta, errors_relative, number of iterations, times
    """

    if np.any(X < 0):
        raise ValueError("X must be non-negative.")

    start_time_init = perf_counter()

    beta2 = beta1 - 1.0

    m, n = X.shape
    W0 = np.random.randn(m, r) if W0 is None else W0
    H0 = np.random.randn(r, n) if H0 is None else H0

    norm_X = np.linalg.norm(X, "fro")
    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    Z = np.zeros((m, n))
    Z[x_is_pos] = X[x_is_pos]

    W, H = W0, H0
    Theta = W @ H
    W_old, H_old = W0.copy(), H0.copy()
    Z_old, Theta_old = Z.copy(), Theta.copy()

    errors = [compute_abs_error(Theta, X) / norm_X]

    if verbose:
        print(
            "Running NMD-T, evolution of [iteration number : relative error in %] - time per iteration"
        )

    initialization_time = perf_counter() - start_time_init
    times = []
    for i in range(0, max_iters):
        start_time_iteration = perf_counter()

        Z = np.minimum(0.0, Theta * x_is_zero)
        Z += X * x_is_pos
        Z = (1 + beta1) * Z - beta1 * Z_old

        # Update of W
        W = np.linalg.lstsq(
            regularization_lambda * np.eye(r) + H_old @ H_old.T, H_old @ Z.T
        )[0].T
        W = (1 + beta2) * W - beta2 * W_old

        # Update of H
        H = np.linalg.lstsq(regularization_lambda * np.eye(r) + W.T @ W, W.T @ Z)[0]
        H = (1 + beta2) * H - beta2 * H_old
        Theta = W @ H

        errors.append(compute_abs_error(Theta, X) / norm_X)
        if errors[-1] < tol:
            times.append(perf_counter() - start_time_iteration)
            if verbose:
                print(f"\nConverged: ||X-max(0,WH)||/||X|| < {tol}")
            break

        if i >= 10 and abs(errors[-1] - errors[-11]) < tol_over_10iters:
            times.append(perf_counter() - start_time_iteration)
            if verbose:
                print(
                    f"\nConverged: abs(rel. err.(i) - rel. err.(i-10)) < {tol_over_10iters}"
                )
            break

        if i < max_iters - 1:
            Theta *= 1.0 + beta1
            Theta -= beta1 * Theta_old

        Z_old, Theta_old, W_old, H_old = Z.copy(), Theta.copy(), W.copy(), H.copy()

        times.append(perf_counter() - start_time_iteration)

        if verbose:
            print(f"[{i} : {(100 * errors[-1]):5f}] - {times[-1]:3f} secs")

    if verbose:
        print_final_msg(times, errors, initialization_time, i)

    return Theta, W, H, errors, i + 1, times


def nmd_3b(
    X: npt.NDArray[np.float_],
    r: int,
    W0: npt.NDArray[np.float_] = None,
    H0: npt.NDArray[np.float_] = None,
    beta1: float = 0.7,
    max_iters: int = 1000,
    tol: float = 1e-4,
    tol_over_10iters: float = 1e-5,
    verbose: bool = True,
) -> (npt.NDArray[np.float_], list[float], int, list[float]):
    """NMD using three-block alternating minimization.

    Args:
        X (npt.NDArray[np.float_]): (m, n) sparse non-negative matrix
        r (int): approximation rank
        W0 (npt.NDArray[np.float_], optional): initial W.
            Defaults to np.random.randn(m, r) if none is provided.
        H0 (npt.NDArray[np.float_], optional): initial H.
            Defaults to np.random.randn(r, n) if none is provided.
        beta1 (float, optional): momentum parameter. Defaults to 0.7.
        max_iters (int, optional): maximum number of iterations. Defaults to 1000.
        tol (float, optional):  stopping criterion on the relative error:
            ||X-max(0,WH)||/||X|| < tol. Defaults to 1e-4.
        tol_over_10iters (float, optional): stopping criterion tolerance on 10
            successive errors: abs(errors[i] - errors[i-10]) < tol_err.
            Defaults to 1e-5.
        verbose (bool, optional): print information to console. Defaults to True.

    Returns:
        (npt.NDArray[np.float_], list[float], int, list[float]):
            Theta, errors_relative, number of iterations, times
    """

    if np.any(X < 0):
        raise ValueError("X must be non-negative.")

    start_time_init = perf_counter()

    m, n = X.shape
    W0 = np.random.randn(m, r) if W0 is None else W0
    H0 = np.random.randn(r, n) if H0 is None else H0

    norm_X = np.linalg.norm(X, "fro")
    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    Z = np.zeros((m, n))
    Z[x_is_pos] = X[x_is_pos]

    W, H = W0, H0
    Theta = W @ H
    Z_old, Theta_old = Z.copy(), Theta.copy()

    errors = [compute_abs_error(Theta, X) / norm_X]

    if verbose:
        print(
            "Running 3B-NMD, evolution of [iteration number : relative error in %] - time per iteration"
        )

    initialization_time = perf_counter() - start_time_init
    times = []
    for i in range(0, max_iters):
        start_time_iteration = perf_counter()

        Z = np.minimum(0.0, Theta * x_is_zero)
        Z += X * x_is_pos
        Z *= 1 + beta1
        Z -= beta1 * Z_old

        # rcond to silence future warning
        # W = np.linalg.lstsq(H @ H.T, H @ Z.T, rcond=None)[0].T
        W = np.linalg.solve(H @ H.T, H @ Z.T).T
        # H = np.linalg.lstsq(W.T @ W, W.T @ Z, rcond=None)[0]
        H = np.linalg.solve(W.T @ W, W.T @ Z)
        Theta = W @ H

        errors.append(compute_abs_error(Theta, X) / norm_X)
        if errors[-1] < tol:
            times.append(perf_counter() - start_time_iteration)
            if verbose:
                print(f"\nConverged: ||X-max(0,WH)||/||X|| < {tol}")
            break

        if i >= 10 and abs(errors[-1] - errors[-11]) < tol_over_10iters:
            times.append(perf_counter() - start_time_iteration)
            if verbose:
                print(
                    f"\nConverged: abs(rel. err.(i) - rel. err.(i-10)) < {tol_over_10iters}"
                )
            break

        if i < max_iters - 1:
            Theta *= 1.0 + beta1
            Theta -= beta1 * Theta_old

        Z_old, Theta_old = Z.copy(), Theta.copy()

        times.append(perf_counter() - start_time_iteration)

        if verbose:
            print(f"[{i} : {(100 * errors[-1]):5f}] - {times[-1]:3f} secs")

    if verbose:
        print_final_msg(times, errors, initialization_time, i)

    return Theta, W, H, errors, i + 1, times


def a_nmd(
    X: npt.NDArray[np.float_],
    r: int,
    Theta0: npt.NDArray[np.float_] = None,
    beta: float = 0.9,
    eta: float = 0.4,
    gamma: float = 1.1,
    gamma_bar: float = 1.05,
    max_iters: int = 1000,
    tol: float = 1.0e-4,
    tol_over_10iters: float = 1.0e-5,
    verbose: bool = True,
) -> (npt.NDArray[np.float_], list[float], int, list[float]):
    """Aggressive Momentum NMD (A-NMD)

    Args:
        X (npt.NDArray[np.float_]): (m, n) sparse non-negative matrix
        r (int): approximation rank
        Theta0 (npt.NDArray[np.float_]): initial Theta. Defaults to
            np.random.randn(m, n) if none is provided.
        beta (float, optional): initial momentum parameter. Defaults to 0.9.
        eta (float, optional): factor that shrinks beta if objective function is
            decreasing. Defaults to 0.4.
        gamma (float, optional): factor that increases beta if objective function
            is decreasing. Defaults to 1.1.
        gamma_bar (float, optional): factor that increases beta bar (upper bound
            for beta) if objective function is decreasing. Defaults to 1.05.
        max_iters (int, optional): maximum number of iterations. Defaults to 1000.
        tol (float, optional):  stopping criterion on the relative error:
            ||X-max(0,WH)||/||X|| < tol. Defaults to 1e-4.
        tol_over_10iters (float, optional): stopping criterion tolerance on 10
            successive errors: abs(errors[i] - errors[i-10]) < tol_err.
                Defaults to 1e-5.
        verbose (bool, optional): print information to console. Defaults to True.

    Returns:
        (npt.NDArray[np.float_], list[float], int, list[float]): Theta,
            errors_relative, number of iterations, times
    """
    if np.any(X < 0):
        raise ValueError("X must be non-negative.")

    ## Code different than paper
    assert (1.0 / eta) > 1.0
    assert (1.0 / eta) > gamma
    assert gamma > gamma_bar

    start_time_init = perf_counter()
    m, n = X.shape
    Theta0 = np.random.randn(m, n) if Theta0 is None else Theta0
    beta_bar = 1.0
    beta_history = [beta]

    x_is_zero = X == 0
    x_is_pos = np.invert(x_is_zero)

    Z0 = np.zeros((m, n))
    Z0[x_is_pos] = X[x_is_pos]

    Z = Z0
    Theta = Theta0
    Z_old = Z0.copy()
    Theta_old = Theta0.copy()

    norm_X = np.linalg.norm(X, ord="fro")
    errors_relative = [compute_abs_error(Theta, X) / norm_X]

    if verbose:
        print(
            "Running A-NMD, evolution of [iteration number : relative error in %] - time per iteration"
        )

    initialization_time = perf_counter() - start_time_init
    times = []
    for i in range(max_iters):
        start_time_iteration = perf_counter()

        Z = np.minimum(0.0, Theta * x_is_zero)
        Z += X * x_is_pos
        Z += beta * (Z - Z_old)

        # U, d, Vt = svds(Z, r)
        # D = np.diag(d)
        # Theta = U @ D @ Vt
        Theta = _find_low_rank_random_truncated(Z, r)

        errors_relative.append(compute_abs_error(Theta, X) / norm_X)

        if errors_relative[-1] < tol:
            if verbose:
                print(f"\nConverged: ||X-max(0,WH)||/||X|| < {tol}")
            break

        if (
            i >= 10
            and abs(errors_relative[-1] - errors_relative[-11]) < tol_over_10iters
        ):
            if verbose:
                print(
                    f"\nConverged: abs(rel. err.(i) - rel. err.(i-10)) < {tol_over_10iters}"
                )
            break

        if i < max_iters - 1:
            Theta += beta * (Theta - Theta_old)

        if i > 1:
            if compute_abs_error(Theta, X) < compute_abs_error(Theta_old, X):
                beta = min(beta_bar, gamma * beta)
                beta_bar = min(
                    1, gamma_bar * beta_bar
                )  # in paper: gamma_bar * beta_bar instead of "gamma_bar * beta"
                beta_history.append(beta)

                Z_old = Z.copy()
                Theta_old = Theta.copy()
            else:
                beta *= eta
                beta_history.append(beta)
                beta_bar = beta_history[i - 2]

                Z = Z_old.copy()
                Theta = Theta_old.copy()

        times.append(perf_counter() - start_time_iteration)

        if verbose:
            print(f"[{i} : {(100 * errors_relative[-1]):5f}] - {times[-1]:3f} secs")

    if verbose:
        print_final_msg(times, errors_relative, initialization_time, i)

    return Theta, errors_relative, i + 1, times


def print_final_msg(times: list[float], errors: list[float], init_time: float, i: int):
    avg_time_per_iter = np.mean(times)
    total_time = init_time + np.sum(times)
    print(f"\nFinal relative error: {100 * errors[-1]}%, after {i + 1} iterations.")
    print(f"Initialization time: {init_time:3f} secs")
    print(f"Mean time per iteration: {avg_time_per_iter:3f} secs")
    print(f"Total time: {total_time:3f} secs\n")


def compute_abs_error(
    Theta: npt.NDArray[np.float_], X: npt.NDArray[np.float_]
) -> float:
    """Compute Frobenius norm of the difference between Theta and X"""
    return np.linalg.norm(np.maximum(0, Theta) - X, ord="fro")


### INITIALIZATION ----
def nuclear_norm_init(
    X: np.ndarray, m: int, n: int, r: int, seed: int, verbose: bool = False
) -> (np.ndarray, np.ndarray):
    rng = np.random.default_rng(seed=seed)
    Theta1 = rng.standard_normal(size=(m, n))
    Theta2, _ = nmd_nuclear_bt(X, Theta1, 3, verbose=verbose)
    ua, sa, va = np.linalg.svd(Theta2, full_matrices=False)
    sa = np.diag(sa)[:r, :r]
    W0 = ua[:, :r]
    H0 = sa @ va[:r, :]
    return W0, H0


def nmd_nuclear_bt(
    X: npt.ArrayLike, Theta: npt.ArrayLike, max_iter: int, verbose: bool = False
) -> (np.ndarray, list[float]):
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


def construct_utility(low_rank_matrix, base_matrix):
    """Construct a utility matrix Z which enforces the invariants of the original
    sparse nonnegative matrix.

    Specifically, it creates Z from a 0-matrix by:
      - Copying the positive elements of the original sparse matrix X into the
        corresponding elements of Z
      - Copying any negative elements of the current low-rank approximation matrix
        L into the corresponding elements of Z, provided those elements of Z
        were not set in the first step
      - Any remaining elements remain 0

    i.e. for each i, j: Z_ij = X is X_ij > 0, else min(0, L_ij).

    Args:
        low_rank_matrix: The current low-rank approximation of the base matrix
        base_matrix: the sparse nonnegative matrix whose low-rank approximation
            is being sought

    Returns:
        A utility matrix whose only positive values are the positive values in
        the base_matrix
    """
    conditions = [base_matrix > 0, low_rank_matrix < 0]
    choices = [base_matrix, low_rank_matrix]
    utility_matrix = np.select(conditions, choices, 0)
    return utility_matrix


def _find_low_rank_random_truncated(utility_matrix, target_rank: int):
    """Compute a low-rank approximation to a matrix via random truncated SVD.

    Args:
        utility_matrix: The matrix to approximate
        target_rank: The target rank of the low-rank approximation. Note that
            in contrast to the full-decomposition method, the underlying algorithm
            will throw an error if the requested rank is equal to or greater than
            the smaller dimension of the matrix (meaning also that this algorithm
            cannot round-trip a full-rank matrix).

    Returns:
        The low-rank approximation.
    """
    svd = TruncatedSVD(n_components=target_rank)
    reduced = svd.fit_transform(utility_matrix)
    low_rank = svd.inverse_transform(reduced)
    return low_rank
