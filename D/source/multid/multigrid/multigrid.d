module multid.multigrid.multigrid;

import std.experimental.logger : logf, infof;
import multid.multigrid.cycle;
import mir.ndslice : Slice;

/++
Method to run some multigrid steps for abstract cycle
+/
Slice!(T*, Dim) multigrid(T, size_t Dim)(Cycle!(T, Dim) cycle, Slice!(T*, Dim) U, size_t iter_cycle, double eps)
{
    foreach (i; 1 .. iter_cycle + 1)
    {

        U = cycle.cycle(U);
        auto norm = cycle.norm(U);
        logf("Residual has a L2-Norm of %f after %d iterations", norm, i);
        if (norm <= eps)
        {
            infof("MG converged after %d iterations with %f error", i, norm);
            break;
        }
    }

    return U;
}

/++
Run some poisson multigrid to solve AU = F with A is a poisson matrix

Params:
    F = Dim-slice
    U = Dim-slice
    level = the depth of the multigrid cycle if it is set to 0, the maxmium depth is choosen
    mu = 1 for V Cycle, 2 for W Cycle, 3 for VW cycle
    iter_cycles = maxium number for cycles
    eps = criteria to stop

Returns: U
+/
Slice!(T*, Dim) poisson_multigrid(T, size_t Dim, uint v1, uint v2)(
        Slice!(T*, Dim) F, Slice!(T*, Dim) U, uint level, uint mu, size_t iter_cycles, double eps = 1e-3)
{
    auto cycle = new PoissonCycle!(T, Dim, v1, v2)(F, mu, level, cast(T)(0));
    return multigrid!(T, Dim)(cycle, U, iter_cycles, eps);
}
