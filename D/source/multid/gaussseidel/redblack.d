module multid.gaussseidel.redblack;

import mir.math: fastmath, approxEqual;
import mir.algorithm.iteration: Chequer, all;
import mir.ndslice : slice, uninitSlice, sliced, Slice, strided;
import multid.gaussseidel.sweep;
import multid.tools.apply_poisson : compute_residual;
import multid.tools.norm : nrmL2;
import std.experimental.logger;
import std.traits : isFloatingPoint;

/++ enum to differentiate between sweep types +/
enum SweepType
{
    ndslice = "ndslice",
    field = "field",
    slice = "slice",
    naive = "naive"
}

/++
This is a Gauss Seidel Red Black implementation
it solves AU = F, with A being a poisson matrix like this
        1 1 1 1 .. 1
        1 4 -1 0 .. 1
        1 -1 4 -1 .. 1
        .          .
        . 0..-1 4 1 .
        1 .. 1 1 1 1
so the borders of U remain unchanged
Params:
    F = slice of dimension Dim
    U = slice of dimension Dim
    R = slice of dimension Dim
    h = the distance between the grid points
Returns: U
+/

Slice!(T*, Dim) GS_RB(SweepType sweeptype = SweepType.ndslice, T, size_t Dim)(
    Slice!(const(T)*, Dim) F,
    Slice!(T*, Dim) U,
    const T h,
    size_t max_iter = 10_000_000,
    size_t norm_iter = 1_000,
    double eps = 1e-8,
    )
    if ((1 <= Dim && Dim <= 8) && isFloatingPoint!T)
{
    auto R = U.shape.slice!T;
    T norm;
    auto it = GS_RB!sweeptype(F, U, R, h, norm, max_iter, norm_iter, eps);
    logf("GS_RB converged after %d iterations with %e error", it, norm);
    return U;
}

@nogc @fastmath
size_t GS_RB(SweepType sweeptype = SweepType.ndslice, T, size_t Dim)(
    Slice!(const(T)*, Dim) F,
    Slice!(T*, Dim) U,
    Slice!(T*, Dim) R, //residual
    const T h,
    out T norm,
    size_t max_iter = 10_000_000,
    size_t norm_iter = 1_000,
    double eps = 1e-8,
    )
    if ((1 <= Dim && Dim <= 8) && isFloatingPoint!T)
{
    mixin("alias sweep = sweep_" ~ sweeptype ~ ";");

    const T h2 = h * h;
    size_t it;
    norm = 0;
    while (it < max_iter)
    {
        it++;
        if (it % norm_iter == 0)
        {
            compute_residual(R, F, U, h);
            norm = R.nrmL2;
            if (norm <= eps)
            {
                break;
            }

        }
        // rote Halbiteration
        sweep(Chequer.red, F, U, h2);
        // schwarze Halbiteration
        sweep(Chequer.black, F, U, h2);
    }
    return it;
}

unittest
{
    const size_t N = 3;
    auto U1 = slice!double([N], 1.0);
    auto F1 = slice!double([N], 0.0);
    F1[1] = 1;
    GS_RB(F1, U1, 1.0, 1);
    assert(U1 == [1.0, 1.0 / 2.0, 1.0].sliced);

    auto U2 = slice!double([N, N], 1.0);
    auto F2 = slice!double([N, N], 0.0);
    F2[1, 1] = 1;

    auto expected = slice!double([N, N], 1.0);
    expected[1, 1] = 3.0 / 4.0;
    GS_RB(F2, U2, 1.0, 1);
    assert(expected == U2);

    auto U3 = slice!double([N, N, N], 1.0);
    auto F3 = slice!double([N, N, N], 0.0);
    F3[1, 1, 1] = 1;
    GS_RB(F3, U3, 1.0, 1);

    auto expected3 = slice!double([N, N, N], 1.0);
    expected3[1, 1, 1] = 5.0;
    expected3[1, 1, 1] *= 1 / 6.0;
    assert(expected3 == U3);

}

unittest
{
    import multid.gaussseidel.sweep;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    const double h2 = 1.0;

    auto U = randomMatrix!(double, 1)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    auto U3 = U.dup;
    const F = slice!double([N], 1.0);

    sweep_naive(Chequer.red, F, U, h2);
    sweep_field(Chequer.red, F, U1, h2);
    sweep_slice(Chequer.red, F, U2, h2);
    sweep_ndslice(Chequer.red, F, U3, h2);
    assert(U == U1);
    assert(U1 == U2);
    assert(all!approxEqual(U, U3));

    sweep_naive(Chequer.black, F, U, h2);
    sweep_field(Chequer.black, F, U1, h2);
    sweep_slice(Chequer.black, F, U2, h2);
    sweep_ndslice(Chequer.black, F, U3, h2);
    assert(U == U1);
    assert(U1 == U2);
    assert(all!approxEqual(U, U3));

}

unittest
{
    import multid.gaussseidel.sweep;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    const double h2 = 1.0;

    auto U = randomMatrix!(double, 2)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    auto U3 = U.dup;
    const F = slice!double([N, N], 1.0);

    sweep_naive(Chequer.red, F, U, h2);
    sweep_field(Chequer.red, F, U1, h2);
    sweep_slice(Chequer.red, F, U2, h2);
    sweep_ndslice(Chequer.red, F, U3, h2);
    assert(U == U1);
    assert(U1 == U2);
    assert(all!approxEqual(U, U3));

    sweep_naive(Chequer.black, F, U, h2);
    sweep_field(Chequer.black, F, U1, h2);
    sweep_slice(Chequer.black, F, U2, h2);
    sweep_ndslice(Chequer.black, F, U3, h2);
    assert(U == U1);
    assert(U1 == U2);
    assert(all!approxEqual(U, U3));
}

unittest
{
    import multid.gaussseidel.sweep;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    auto U = randomMatrix!(double, 3)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    auto U3 = U.dup;
    const F = slice!double([N, N, N], 1.0);
    const double h2 = 1.0;

    sweep_naive(Chequer.red, F, U, h2);
    sweep_field(Chequer.red, F, U1, h2);
    sweep_slice(Chequer.red, F, U2, h2);
    sweep_ndslice(Chequer.red, F, U3, h2);
    // import std.stdio;
    // writeln(U - U1);
    assert(U == U1);
    assert(U1 == U2);
    assert(all!approxEqual(U, U3));

    sweep_naive(Chequer.black, F, U, h2);
    sweep_field(Chequer.black, F, U1, h2);
    sweep_slice(Chequer.black, F, U2, h2);
    sweep_ndslice(Chequer.black, F, U3, h2);
    assert(U == U1);
    assert(U1 == U2);
    assert(all!approxEqual(U, U3));
}
