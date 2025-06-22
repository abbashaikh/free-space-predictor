'''Helper Functions'''
import warnings

import numpy as np

####### To Calculate Number of Samples #######
def bisection(N_low, N_high, sample_function): 
    """
    :param N_low: Low guess for the intersection
    :param N_high: High guess for the intersection
    :param sample_function: Function for which to find the zero crossing
    :return: The zero crossing
    """
    # print(f"Running bisection to find S between {N_low} and {N_high}...", end="")

    a = N_low
    b = N_high
    f = sample_function

    TOL = 1e-9

    if a > b:
        print('Error: a > b!')
        return -1

    value_a = f(a)

    for _ in range(1000):
        c = (a + b) / 2.
        value_c = f(c)

        if value_c == 0 or (b - a) / 2. < TOL:
            # print(f" Found {c:.2f}")
            return c

        if np.sign(value_c) == np.sign(value_a):
            a = c
            value_a = value_c
        else:
            b = c

    print("Bisection failed to converge!")


def nchoosek_rooted(n, k, root):
    """
    To avoid numerical errors, an implementation of N choose K where the root is resolved in between
    :return: (n choose k)^(1/root)
    """
    y = 1.0
    for i in range(k):
        y = y * ((n - (k - i - 1.)) / (k - i))**(1. / root)
    return y


def compute_risk(S, confidence, support):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1. - ((confidence / S) ** (1. / (S - support))) \
                * (1. / nchoosek_rooted(S, support, S - support))


def find_sample_size(support, risk, confidence):
    risk_function = lambda S, support: compute_risk(S, confidence, support)      # Scenario Optimization guarantee for any support
    max_risk_function = lambda S: risk - risk_function(S, support)               # "" for the specified support limit

    S_low = support+1
    S_high = 500000

    S_double = bisection(S_low, S_high, max_risk_function)
    S = np.floor(S_double) + 1

    return S


def is_within_distance(
        arr1: np.ndarray,
        arr2: np.ndarray,
        distance: float):
    return np.linalg.norm(arr1 - arr2) <= distance


def get_constraint_coeffs(
    ego_pos: np.ndarray,    # shape (2,)
    agent_pos: np.ndarray,  # shape (T, M, 2)
    r: float
) -> np.ndarray:
    """
    Computes (a, b, c) for the half-space a x + b y + c >= 0
    between the ego agent and M agents over T time steps, with NaN handling.

    Args:
      ego_pos:   (2,) array
      agent_pos: (T, M, 2) array of agent positions
      r:         safety radius

    Returns:
      coeffs: (T, M, 3) array of [a, b, c] coefficients;
              rows where agent_pos was NaN are zeroed out.
    """
    # enforce 3D input
    if agent_pos.ndim != 3 or agent_pos.shape[2] != 2:
        raise ValueError("agent_pos must have shape (T, M, 2)")

    T, M, _ = agent_pos.shape

    # mask out any NaNs per (t, m)
    nan_mask = np.isnan(agent_pos).any(axis=2)  # (T, M)

    # broadcast ego_pos to (T, M, 2)
    ego = ego_pos.reshape(1, 1, 2)

    # vector from each agent to ego: u[t,m] = ego - agent_pos[t,m]
    u = ego - agent_pos                          # (T, M, 2)
    norms = np.linalg.norm(u, axis=2, keepdims=True)  # (T, M, 1)
    A = u / norms                                # (T, M, 2)

    # compute c = r + A_x*x + A_y*y
    c = r + A[...,0]*agent_pos[...,0] + A[...,1]*agent_pos[...,1]  # (T, M)

    # pack into (T, M, 3)
    coeffs = np.empty((T, M, 3), dtype=A.dtype)
    coeffs[..., 0] = A[..., 0]
    coeffs[..., 1] = A[..., 1]
    coeffs[..., 2] = -c

    # zero-out entries where agent_pos was NaN
    coeffs[nan_mask, :] = 0

    return coeffs


def check_symmetric(matrix):
    """
    Checks if the given matrix is symmetric (A == A^T).
    Raises a ValueError if it is not symmetric.
    """
    if not np.allclose(matrix, matrix.T):  # Checks if A == A^T (within numerical tolerance)
        raise ValueError("Matrix is not symmetric!")


def transform_coords_np(coords: np.ndarray, tf: np.ndarray,) -> np.ndarray:
    """
    coords:    np.ndarray of shape (M, N, 2)
    transforms: np.ndarray of shape (M, 3, 3)
    returns:   np.ndarray of shape (M, N, 2)
    
    Applies each 3×3 homogeneous transform to the corresponding set of 2D points.
    """
    M, N, _ = coords.shape

    # Lift to homogeneous (M, N, 3)
    ones = np.ones((M, N, 1), dtype=coords.dtype)
    coords_h = np.concatenate([coords, ones], axis=-1)

    # Batch-matmul: for each m, transforms[m] @ coords_h[m].T
    #    - coords_h.transpose(0,2,1) is shape (M, 3, N)
    #    - result is (M, 3, N)
    transformed_h = tf @ coords_h.transpose(0, 2, 1)

    # Back to (M, N, 3), then drop the homogeneous coordinate
    transformed = transformed_h.transpose(0, 2, 1)[..., :2]
    return transformed
