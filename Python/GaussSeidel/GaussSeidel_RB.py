import numpy as np


def GS_RB(F, U=None, max_iter=1000, eps=1e-10):
    """Implementation of Gauss Seidl Red Black iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """

    if U is None:
        U = np.zeros_like(F)

    if len(F.shape) == 1:
        # do the sweep
        sweep = sweep_1D
    elif len(F.shape) == 2:
        # do the sweep
        sweep = sweep_2D
    elif len(F.shape) == 3:
        # Anzahl an Gauss-Seidel-Iterationen ausfuehren
        sweep = sweep_3D
    else:
        raise ValueError("Wrong Shape!!!")

    # last_U = None
    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        sweep(1, F, U)
        # schwarze Halbiteration
        sweep(0, F, U)

        # pruefe Abbruchkriterium
        # if last_U is not None and np.linalg.norm(U - last_U) < eps:
        #     return U

        # last_U = U
    return U


# --- 1D Fall ---
def sweep_1D(color, F, U):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    """
    n = F.shape[0]

    for i in range(1, n - 1):
        if i % 2 == color:
            U[i] = (U[i - 1] +
                    U[i + 1] -
                    F[i]) / 2.0
# ----------------

# --- 2D Fall ---


def sweep_2D(color, F, U):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    """

    m, n = F.shape

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            if (i + j) % 2 == color:
                U[i, j] = (U[i - 1, j] +
                           U[i + 1, j] +
                           U[i, j - 1] +
                           U[i, j + 1] -
                           F[i, j]) / 4.0
# ----------------

# --- 3D Fall ---


def sweep_3D(color, F, U):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    """

    m, n, o = F.shape

    for k in range(1, o - 1):
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                if (i + j + k) % 2 == color:
                    U[i, j, k] = (U[i - 1, j, k] +
                                  U[i + 1, j, k] +
                                  U[i, j - 1, k] +
                                  U[i, j + 1, k] +
                                  U[i, j, k - 1] +
                                  U[i, j, k + 1] -
                                  F[i, j, k]) / 6.0
