import numpy as np
from abc import abstractmethod
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..tools.operators import poisson_operator_like
from ..tools.apply_poisson import apply_poisson

from .restriction import restriction, weighted_restriction
from .prolongation import prolongation


class AbstractCycle:
    def __init__(self, F, v1, v2, mu, l, eps=1e-8):
        self.v1 = v1
        self.v2 = v2
        self.mu = mu
        self.F = F
        self.l = l
        self.eps = eps
        self.h = 1 / F.shape[0]
        if (self.l == 0):
            self.l = int(np.log2(self.F.shape[0])) - 1
        # ceck if l is plausible
        if np.log2(self.F.shape[0]) < self.l:
            raise ValueError('false value of levels')

    def __call__(self, U, h=None):
        return self.do_cycle(self.F, U, self.l, h)

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

    @abstractmethod
    def norm(self, U):
        pass

    @abstractmethod
    def restriction(self, r):
        pass

    def _residual(self, U):
        return self._compute_residual(self.F, U, self.h)

    def _compute_correction(self, r, l, h):
        e = np.zeros_like(r)
        for _ in range(self.mu):
            e = self.do_cycle(r, e, l, h)
        return e

    def do_cycle(self, F, U, l, h=None):
        if h is None:
            h = 1 / U.shape[0]

        if l <= 1 or U.shape[0] <= 1:
            return self._solve(F, U, h)

        U = self._presmooth(F=F, U=U, h=h)

        r = self._compute_residual(F=F, U=U, h=2 * h)

        r = self.restriction(r)

        e = self._compute_correction(r, l - 1, 2 * h)

        e = prolongation(e, U.shape)

        # correction
        U += e

        return self._postsmooth(F=F, U=U, h=h)


class PoissonCycle(AbstractCycle):
    def __init__(self, F, v1, v2, mu, l, eps=1e-8):
        super().__init__(F, v1, v2, mu, l, eps)

    def _presmooth(self, F, U, h=None):
        return GS_RB(
            F,
            U=U,
            h=h,
            max_iter=self.v1,
            eps=self.eps)

    def _postsmooth(self, F, U, h=None):
        return GS_RB(
            F,
            U=U,
            h=h,
            max_iter=self.v2,
            eps=self.eps)

    def _compute_residual(self, F, U, h):
        return F - apply_poisson(U, h)

    def _solve(self, F, U, h):
        return GS_RB(
            F=F,
            U=U,
            h=h,
            max_iter=100_000,
            eps=self.eps,
            norm_iter=5)

    def norm(self, U):
        residual = self._residual(U)
        return np.linalg.norm(residual)

    def restriction(self, r):
        return weighted_restriction(r)
