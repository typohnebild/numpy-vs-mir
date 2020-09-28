module multid.gaussseidel.redblack;

import multid.tools.apply_poisson : compute_residual;
import multid.tools.norm : nrmL2;

import mir.ndslice : slice, sliced, Slice, strided;
import std.traits : isFloatingPoint;

import multid.gaussseidel.sweep;
import std.experimental.logger;

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
    U = slice of dimension Dim
    F = slice of dimension Dim
    h = the distance between the grid points
Returns: U
+/
Slice!(T*, Dim) GS_RB(T, size_t Dim, size_t max_iter = 10_000_000, size_t norm_iter = 1_000,
        double eps = 1e-8, SweepType sweeptype = SweepType.field)(in Slice!(T*, Dim) F, Slice!(T*, Dim) U, in T h)
        if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
{
    mixin("alias sweep = sweep_" ~ sweeptype ~ ";");

    const T h2 = h * h;

    foreach (it; 1 .. max_iter + 1)
    {
        if (it % norm_iter == 0)
        {
            const auto norm = compute_residual!(T, Dim)(F, U, h).nrmL2;
            if (norm <= eps)
            {
                logf("GS_RB converged after %d iterations with %e error", it, norm);
                break;
            }

        }
        // rote Halbiteration
        sweep!(T, Dim, Color.red)(F, U, h2);
        // schwarze Halbiteration
        sweep!(T, Dim, Color.black)(F, U, h2);
    }
    return U;
}

unittest
{
    const size_t N = 3;
    auto U1 = slice!double([N], 1.0);
    auto F1 = slice!double([N], 0.0);
    F1[1] = 1;
    GS_RB!(double, 1, 1)(F1, U1, 1.0);
    assert(U1 == [1.0, 1.0 / 2.0, 1.0].sliced);

    auto U2 = slice!double([N, N], 1.0);
    auto F2 = slice!double([N, N], 0.0);
    F2[1, 1] = 1;

    auto expected = slice!double([N, N], 1.0);
    expected[1, 1] = 3.0 / 4.0;
    GS_RB!(double, 2, 1)(F2, U2, 1.0);
    assert(expected == U2);

    auto U3 = slice!double([N, N, N], 1.0);
    auto F3 = slice!double([N, N, N], 0.0);
    F3[1, 1, 1] = 1;
    GS_RB!(double, 3, 1)(F3, U3, 1.0);

    auto expected3 = slice!double([N, N, N], 1.0);
    expected3[1, 1, 1] = 5.0 / 6.0;
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

    sweep_naive!(double, 1, Color.red)(F, U, h2);
    sweep_field!(double, 1, Color.red)(F, U1, h2);
    sweep_slice!(double, 1, Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(double, 1, Color.black)(F, U, h2);
    sweep_field!(double, 1, Color.black)(F, U1, h2);
    sweep_slice!(double, 1, Color.black)(F, U2, h2);
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

    sweep_naive!(double, 2, Color.red)(F, U, h2);
    sweep_field!(double, 2, Color.red)(F, U1, h2);
    sweep_slice!(double, 2, Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(double, 2, Color.black)(F, U, h2);
    sweep_field!(double, 2, Color.black)(F, U1, h2);
    sweep_slice!(double, 2, Color.black)(F, U2, h2);
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

    sweep_naive!(double, 3, Color.red)(F, U, h2);
    sweep_field!(double, 3, Color.red)(F, U1, h2);
    sweep_slice!(double, 3, Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(double, 3, Color.black)(F, U, h2);
    sweep_field!(double, 3, Color.black)(F, U1, h2);
    sweep_slice!(double, 3, Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);
}
