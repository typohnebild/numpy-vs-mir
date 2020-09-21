module multid.gaussseidel.redblack;

import multid.tools.apply_poisson : compute_residual;
import multid.tools.norm : nrmL2;

import mir.ndslice : slice, sliced, Slice, strided;
import std.traits : isFloatingPoint;

import std.stdio : writeln;
import multid.gaussseidel.slow_sweep;

/++
    red is for even indicies
    black for the odd
+/
enum Color
{
    red = 1u,
    black = 0u
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
Slice!(T*, Dim) GS_RB(T, size_t Dim, size_t max_iter = 10_000_000, size_t norm_iter = 1_000, double eps = 1e-8)(
        in Slice!(T*, Dim) F, Slice!(T*, Dim) U, in T h)
        if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
{
    const T h2 = h * h;

    foreach (it; 1 .. max_iter + 1)
    {
        if (it % norm_iter == 0)
        {
            const auto norm = compute_residual!(T, Dim)(F, U, h).nrmL2;
            if (norm <= eps)
            {
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

/++
This is a sweep implementation for 1D
    it calculates U[i] = (U[i-1] + U[i+1])/2
    for every cell except the borders
Params:
    F  = slice of dimension Dim
    U  = slice of dimension Dim
    h2 = the squared distance between the grid points
+/
void sweep(T, size_t Dim : 1, Color color)(in Slice!(T*, 1) F, Slice!(T*, 1) U, in T h2)
{
    const auto N = F.shape[0];
    auto UF = U.field;
    auto FF = F.field;
    for (size_t i = 2u - color; i < N - 1u; i += 2u)
    {
        UF[i] = (UF[i - 1u] + UF[i + 1u] - FF[i] * h2) / 2.0;
    }
}

/++
This is a sweep implementation for 2D
    it calculates U[i,j] = (U[i-1, j] + U[i+1, j] + U[i, j-1] +U[i, j+1] - h2 * F[i,j])/4
    for every cell except the borders
Params:
    F  = slice of dimension Dim
    U  = slice of dimension Dim
    h2 = the squared distance between the grid points
+/
void sweep(T, size_t Dim : 2, Color color)(in Slice!(T*, 2) F, Slice!(T*, 2) U, in T h2)
{
    const auto m = F.shape[0];
    const auto n = F.shape[1];
    auto UF = U.field;
    auto FF = F.field;

    foreach (i; 1 .. m - 1)
    {
        const flatrow = i * m;
        for (size_t j = 1 + (i + 1 + color) % 2; j < n - 1; j += 2)
        {
            const flatindex = flatrow + j;
            UF[flatindex] = (
                    UF[flatindex - m] +
                    UF[flatindex + m] +
                    UF[flatindex - 1] +
                    UF[flatindex + 1] - h2 * FF[flatindex]) / cast(T) 4;
        }
    }
}

/++
This is a sweep implementation for 3D
    it calculates U[i,j,k] = (U[i-1,j,k] + U[i+1,j,k] + U[i,j-1,k] +U[i,j+1,k] ... - h2 * F[i,j,k])/4
    for every cell except the borders
Params:
    F  = slice of dimension Dim
    U  = slice of dimension Dim
    h2 = the squared distance between the grid points
+/
void sweep(T, size_t Dim : 3, Color color)(in Slice!(T*, 3) F, Slice!(T*, 3) U, in T h2)
{
    const auto m = F.shape[0];
    const auto n = F.shape[1];
    const auto l = F.shape[2];
    auto UF = U.field;
    auto FF = F.field;
    foreach (i; 1 .. m - 1)
    {
        foreach (j; 1 .. n - 1)
        {
            const auto flatindex2d = i * (n * l) + j * l;
            for (size_t k = 1u + (i + j + 1 + color) % 2; k < l - 1u; k += 2)
            {
                const flatindex = flatindex2d + k;
                UF[flatindex] = (
                        UF[flatindex - n * l] +
                        UF[flatindex + n * l] +
                        UF[flatindex - l] +
                        UF[flatindex + l] +
                        UF[flatindex - 1] +
                        UF[flatindex + 1] - h2 * FF[flatindex]) / 6.0;

            }
        }
    }
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
    import multid.gaussseidel.slow_sweep : slow_sweep, sweep_naive;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    const double h2 = 1.0;

    auto U = randomMatrix!(double, 1)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    const auto F = slice!double([N], 1.0);

    sweep_naive!(double, 1, Color.red)(F, U, h2);
    sweep!(double, 1, Color.red)(F, U1, h2);
    slow_sweep!(double, 1, Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(double, 1, Color.black)(F, U, h2);
    sweep!(double, 1, Color.black)(F, U1, h2);
    slow_sweep!(double, 1, Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

}

unittest
{
    import multid.gaussseidel.slow_sweep : sweep_naive, slow_sweep;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    const double h2 = 1.0;

    auto U = randomMatrix!(double, 2)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    const auto F = slice!double([N, N], 1.0);

    sweep_naive!(double, 2, Color.red)(F, U, h2);
    sweep!(double, 2, Color.red)(F, U1, h2);
    slow_sweep!(double, 2, Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(double, 2, Color.black)(F, U, h2);
    sweep!(double, 2, Color.black)(F, U1, h2);
    slow_sweep!(double, 2, Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);
}

unittest
{
    import multid.gaussseidel.slow_sweep : sweep_naive, slow_sweep;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    auto U = randomMatrix!(double, 3)(N);
    auto U1 = U.dup;
    auto U2 = U.dup;
    const auto F = slice!double([N, N, N], 1.0);
    const double h2 = 1.0;

    sweep_naive!(double, 3, Color.red)(F, U, h2);
    sweep!(double, 3, Color.red)(F, U1, h2);
    slow_sweep!(double, 3, Color.red)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

    sweep_naive!(double, 3, Color.black)(F, U, h2);
    sweep!(double, 3, Color.black)(F, U1, h2);
    slow_sweep!(double, 3, Color.black)(F, U2, h2);
    assert(U == U1);
    assert(U1 == U2);

}
