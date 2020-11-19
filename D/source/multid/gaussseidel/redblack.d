module multid.gaussseidel.redblack;

import mir.math: fastmath;
import mir.ndslice : slice, uninitSlice, sliced, Slice, strided;
import multid.gaussseidel.sweep;
import multid.tools.apply_poisson : compute_residual;
import multid.tools.norm : nrmL2;
import std.experimental.logger;
import std.traits : isFloatingPoint;

/++
    red is for even indicies
    black for the odd
+/
enum Color
{
    red = 1u,
    black = 0u
}

/++ enum to differentiate between sweep types +/
enum SweepType
{
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

Slice!(T*, Dim) GS_RB(size_t max_iter = 10_000_000, size_t norm_iter = 1_000,
        double eps = 1e-8, SweepType sweeptype = SweepType.field, T, size_t Dim)(
            Slice!(const(T)*, Dim) F,
            Slice!(T*, Dim) U,
            const T h)
        if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
{
    auto R = U.shape.slice!T;
    T norm;
    auto it = GS_RB!(max_iter, norm_iter, eps, sweeptype)(F, U, R, h, norm);
    logf("GS_RB converged after %d iterations with %e error", it, norm);
    return U;
}

@nogc @fastmath
size_t GS_RB(size_t max_iter = 10_000_000, size_t norm_iter = 1_000,
        double eps = 1e-8, SweepType sweeptype = SweepType.field, T, size_t Dim)(
            Slice!(const(T)*, Dim) F,
            Slice!(T*, Dim) U,
            Slice!(T*, Dim) R, //residual
            const T h,
            out T norm)
        if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
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
        sweep!(Color.red)(F, U, h2);
        // schwarze Halbiteration
        sweep!(Color.black)(F, U, h2);
    }
    return it;
}

unittest
{
    const size_t N = 3;
    auto U1 = slice!double([N], 1.0);
    auto F1 = slice!double([N], 0.0);
    F1[1] = 1;
    GS_RB!(1)(F1, U1, 1.0);
    assert(U1 == [1.0, 1.0 / 2.0, 1.0].sliced);

    auto U2 = slice!double([N, N], 1.0);
    auto F2 = slice!double([N, N], 0.0);
    F2[1, 1] = 1;

    auto expected = slice!double([N, N], 1.0);
    expected[1, 1] = 3.0 / 4.0;
    GS_RB!(1)(F2, U2, 1.0);
    assert(expected == U2);

    auto U3 = slice!double([N, N, N], 1.0);
    auto F3 = slice!double([N, N, N], 0.0);
    F3[1, 1, 1] = 1;
    GS_RB!(1)(F3, U3, 1.0);

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
    const auto F = slice!double([N], 1.0);

    sweep_naive!(Color.red)(F, U, h2);
    sweep_field!(Color.red)(F, U1, h2);
    sweep_slice!(Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(Color.black)(F, U, h2);
    sweep_field!(Color.black)(F, U1, h2);
    sweep_slice!(Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

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
    const auto F = slice!double([N, N], 1.0);

    sweep_naive!(Color.red)(F, U, h2);
    sweep_field!(Color.red)(F, U1, h2);
    sweep_slice!(Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(Color.black)(F, U, h2);
    sweep_field!(Color.black)(F, U1, h2);
    sweep_slice!(Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);
}

unittest
{
    import multid.gaussseidel.sweep;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    auto U = randomMatrix!(double, 3)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    const auto F = slice!double([N, N, N], 1.0);
    const double h2 = 1.0;

    sweep_naive!(Color.red)(F, U, h2);
    sweep_field!(Color.red)(F, U1, h2);
    sweep_slice!(Color.red)(F, U2, h2);
    // import std.stdio;
    // writeln(U - U1);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(Color.black)(F, U, h2);
    sweep_field!(Color.black)(F, U1, h2);
    sweep_slice!(Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);
}
