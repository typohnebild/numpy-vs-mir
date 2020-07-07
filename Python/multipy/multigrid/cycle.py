import numpy as np
from abc import abstractmethod
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..tools.operators import poisson_operator_like
from ..tools.apply_poisson import apply_poisson

from .restriction import restriction
from .prolongation import prolongation


class AbstractCycle:
    def __init__(self, F, v1, v2, mu):
        self.v1 = v1
        self.v2 = v2
        self.mu = mu
        self.F = F
        self.eps = 1e-30
        self.h = 1 / F.shape[0]

    def __call__(self, U, l, h=None):
        return self.do_cycle(self.F, U, l, h)

    @abstractmethod
    def _presmooth(self, F, U, h):
        pass

    @abstractmethod
    def _postsmooth(self, F, U, h):
        pass

    @abstractmethod
    def _compute_residual(self, F, U, h):
        pass

    @abstractmethod
    def _solve(self, F, U, h):
        pass

    def residual(self, U):
        return self._compute_residual(self.F, U, self.h)

    def _compute_correction(self, r, l, h):
        e = np.zeros_like(r)
        for _ in range(self.mu):
            e = self.do_cycle(r.copy(), e, l, h)
        return e

    def do_cycle(self, F, U, l, h=None):
        if h is None:
            h = 1 / U.shape[0]

        if l <= 1 or U.shape[0] <= 1:
            return self._solve(F, U, h)

        U = self._presmooth(F=F, U=U, h=h)

        r = self._compute_residual(F=F, U=U, h=2 * h)

        r = restriction(r)

        e = self._compute_correction(r, l - 1, 2 * h)

        e = prolongation(e, U.shape)

        # correction
        U = U + e

        return self._postsmooth(F=F, U=U, h=h)


class PoissonCycle(AbstractCycle):
    def __init__(self, F, v1, v2, mu):
        super().__init__(F, v1, v2, mu)

    def _presmooth(self, F, U, h=None):
        return GS_RB(F, U=U, h=h, max_iter=self.v1, eps=self.eps)

    def _postsmooth(self, F, U, h=None):
        return GS_RB(F, U=U, h=h, max_iter=self.v2, eps=self.eps)

    def _compute_residual(self, F, U, h):
        return F - apply_poisson(U, h)

    def _solve(self, F, U, h):
        return GS_RB(F=F, U=U, h=h, max_iter=5_000, eps=1e-3)


# TODO atm it is not really general because of the poisson operator calls
# and in this version it will not work i guess because of the missing h
class GeneralCycle(AbstractCycle):
    def __init__(self, A, F, v1, v2, mu):
        super().__init__(F, v1, v2, mu)
        self.A = A

    def _presmooth(self, F, U):
        A = poisson_operator_like(F)
        return gauss_seidel(A, F, U, max_iter=self.v1)

    def _postsmooth(self, F, U):
        A = poisson_operator_like(F)
        return gauss_seidel(A, F, U, max_iter=self.v2)

    def _compute_residual(self, F, U):
        A = poisson_operator_like(F)
        return F - (A @ U)

    def _solve(self, F, U, h):
        A = poisson_operator_like(F)
        return np.linalg.solve(A, F)
