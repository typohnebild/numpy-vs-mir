module multid.multigrid.multigrid;

import std.experimental.logger : logf, infof;
import multid.multigrid.cycle;
import mir.ndslice : Slice;

import multid.gaussseidel.redblack : SweepType;

/++
Method to run some multigrid steps for abstract cycle
+/
Slice!(T*, Dim) multigrid(T, size_t Dim)(Cycle!(T, Dim) cycle, Slice!(T*, Dim) U, size_t iter_cycle, double eps)
{
    //scale the epsilon with the number of gridpoints
    eps *= U.shape[0] * U.shape[0];
    foreach (i; 1 .. iter_cycle + 1)
    {

        U = cycle.cycle(U);
        auto norm = cycle.norm(U);
        logf("Residual has a L2-Norm of %f after %d iterations", norm, i);
        if (norm <= eps)
        {
            infof("MG converged after %d iterations with %e error", i, norm);
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
        Slice!(T*, Dim) F, Slice!(T*, Dim) U, uint level, uint mu, size_t iter_cycles,
        string sweep = "field", T eps = 1e-6, T h = 0)
{
    Cycle!(T, Dim) cycle;
    switch (sweep)
    {
    case "slice":
        cycle = new PoissonCycle!(T, Dim, v1, v2, SweepType.slice)(F, mu, level, h);
        break;
    case "naive":
        cycle = new PoissonCycle!(T, Dim, v1, v2, SweepType.naive)(F, mu, level, h);
        break;
    default:
        cycle = new PoissonCycle!(T, Dim, v1, v2, SweepType.field)(F, mu, level, h);
    }
    return multigrid!(T, Dim)(cycle, U, iter_cycles, eps);
}

unittest
{

    import multid.tools.util : randomMatrix;
    import multid.gaussseidel.redblack : GS_RB;
    import mir.ndslice : slice;
    import std.experimental.logger : globalLogLevel, LogLevel;

    globalLogLevel(LogLevel.off);

    const size_t N = 50;
    immutable h = 1.0 / N;

    auto U = randomMatrix!(double, 2)(N);

    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    F[0][0 .. $] = 1.0;
    F[1 .. $, 0] = 1.0;
    F[$ - 1][1 .. $] = 0.0;
    F[1 .. $, $ - 1] = 0.0;
    auto U1 = U.dup;
    poisson_multigrid!(double, 2, 2, 2)(F, U, 0, 2, 100, "field", 1e-9);

    GS_RB!(double, 2)(F, U1, h);

    import numir : approxEqual;

    assert(approxEqual(U, U1, 1e-8));

}
