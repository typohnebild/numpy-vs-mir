import numpy as np
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..tools.operators import poisson_operator_like
from ..tools.apply_poisson import apply_poisson

from .restriction import restriction
from .prolongation import prolongation


class Cycle:
    def __init__(self, v1, v2, mu):
        self.v1 = v1
        self.v2 = v2
        self.mu = mu
        self.eps = 1e-30

    def __call__(self, F, U, l, h=None):
        return self.do_cycle(F, U, l, h)

    def _presmooth(self, F, U, h=None):
        return GS_RB(F, U=U, h=h, max_iter=self.v1, eps=self.eps)

    def _postsmooth(self, F, U, h=None):
        return GS_RB(F, U=U, h=h, max_iter=self.v2, eps=self.eps)

    def _compute_residual(self, F, U, h):
        return F - apply_poisson(U, 2 * h)

    def _compute_correction(self, r, l, h):
        e = np.zeros_like(r)
        for _ in range(self.mu):
            e = self.do_cycle(r, e, l, h)
        return e

    def do_cycle(self, F, U, l, h=None):
        if h is None:
            h = 1 / U.shape[0]

        if l <= 1 or U.shape[0] <= 1:
            # solve
            return GS_RB(F=F, U=U, h=h, max_iter=5000)

        U = self._presmooth(F=F, U=U, h=h)

        r = self._compute_residual(F=F, U=U, h=h)

        r = restriction(r)

        e = self._compute_correction(r, l - 1, 2 * h)

        # prolongation
        e = prolongation(e, U.shape)

        # correction
        U = U + e

        # post smoothing
        return self._postsmooth(F=F, U=U, h=h)
