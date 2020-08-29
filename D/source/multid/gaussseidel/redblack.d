module multid.gaussseidel.redblack;

import multid.tools.apply_poisson;
import multid.tools.norm : nrmL2;

import mir.ndslice;
import std.traits : isFloatingPoint;

import std.stdio : writeln;
import pretty_array;

/++
    red is for even indicies
    black for the odd
+/
enum Color
{
    red = 0u,
    black = 1u
}

/++
This is a Gauss Seidel Red Black implementation for 1D
+/
Slice!(T*, Dim) GS_RB(T, size_t Dim, size_t max_iter = 10_000_000,
        size_t norm_iter = 10_000, double eps = 1e-8)(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T h)
        if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
{

    const T h2 = h * h;

    foreach (it; 0 .. max_iter)
    {
        if (it % norm_iter == 0)
        {
            const auto norm = residual_norm!(T, Dim)(F, U, h);
            if (norm <= eps)
            {
                it.writeln;
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
+/
void sweep(T, size_t Dim : 1, Color color)(const Slice!(T*, 1) F, Slice!(T*, 1) U, T h2)
{
    U[1 + color .. $ - 1].strided!0(2)[] = (
            U[0 + color .. $ - 2].strided!0(2) + U[2 + color .. $].strided!0(
            2) - h2 * F[1 + color .. $ - 1].strided!0(2)) / 2.0;
}

/++
This is a sweep implementation for 2D
+/
void sweep(T, size_t Dim : 2, Color color)(const Slice!(T*, 2) F, Slice!(T*, 2) U, T h2)
{
    const auto m = F.shape[0];
    const auto n = F.shape[1];
    static if (color == Color.red)
    {
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] = U[0 .. m - 2, 1 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m, 1 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[1 .. m - 1, 0 .. n - 2].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[1 .. m - 1, 2 .. n].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] -= F[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2) * h2 / 4.0;

        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] = U[1 .. m - 2, 2 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[3 .. m, 2 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m - 1, 1 .. n - 2].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m - 1, 3 .. n].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] -= F[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2) * h2 / 4.0;
    }
    else static if (color == Color.black)
    {
        U[1 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] = U[0 .. m - 2, 2 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m, 2 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[1 .. m - 1, 1 .. n - 2].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[1 .. m - 1, 3 .. n].strided!(0, 1)(2, 2) / 4.0;
        U[1 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] -= F[1 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2) * h2 / 4.0;

        U[2 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] = U[1 .. m - 2, 1 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[3 .. m, 1 .. n - 1].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m - 1, 0 .. n - 2].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m - 1, 2 .. n].strided!(0, 1)(2, 2) / 4.0;
        U[2 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] -= F[2 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2) * h2 / 4.0;
    }
    else
    {
        static assert(false, color.stringof ~ "invalid color");
    }
}

/++
This is a sweep implementation for 3D
+/
void sweep(T, size_t Dim : 3, Color color)(const Slice!(T*, 3) F, Slice!(T*, 3) U, T h2)
{
    const auto n = F.shape[0];
    const auto m = F.shape[1];
    const auto l = F.shape[1];
    for (size_t i = 1u; i < n - 1u; i++)
    {
        for (size_t j = 1u; j < m - 1u; j++)
        {
            for (size_t k = 1u; k < l - 1u; k++)
            {
                if ((i + j + k) % 2 == color)
                {
                    U[i, j, k] = (U[i - 1, j, k] + U[i + 1, j, k] + U[i, j - 1,
                            k] + U[i, j + 1, k] + U[i, j, k - 1] + U[i, j, k + 1] - h2 * F[i, j, k]) / 6.0;

                }
            }
        }
    }
}

/++
    Computes the L2 norm of the residual
+/
T residual_norm(T, size_t Dim)(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T h)
{
    import mir.math.sum : sum;

    static if (Dim == 1)
    {
        return nrmL2((F - apply_poisson!(T, Dim)(U, h))[1 .. $ - 1]);
    }
    else static if (Dim == 2)
    {
        return nrmL2((F - apply_poisson!(T, Dim)(U, h))[1 .. $ - 1, 1 .. $ - 1]);
    }
    else static if (Dim == 3)
    {
        return nrmL2((F - apply_poisson!(T, Dim)(U, h))[1 .. $ - 1, 1 .. $ - 1, 1 .. $ - 1]);
    }
    else
    {
        static assert(false, Dim.stringof ~ " is not a supported dimension!");
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
    import std.range : generate;
    import std.random : uniform;
    import std.algorithm : fill;

    const size_t N = 10;
    auto U = slice!double([N], 1.0);
    U.field.fill(generate!(() => uniform(0.0, 1.0)));
    auto U1 = U.dup;
    const auto F = slice!double([N], 0.0);
    const double h2 = 1.0;
    for (size_t i = 1u + Color.red; i < N - 1u; i += 2u)
    {
        U[i] = (U[i - 1u] + U[i + 1u] - F[i] * h2) / 2.0;
    }
    sweep!(double, 1, Color.red)(F, U1, h2);
    assert(U == U1);

    for (size_t i = 1u + Color.black; i < N - 1u; i += 2u)
    {
        U[i] = (U[i - 1u] + U[i + 1u] - F[i] * h2) / 2.0;
    }
    sweep!(double, 1, Color.black)(F, U1, h2);
    assert(U == U1);

}

unittest
{
    import std.range : generate;
    import std.random : uniform;
    import std.algorithm : fill;

    const size_t N = 10;
    auto U = slice!double([N, N], 1.0);
    U.field.fill(generate!(() => uniform(0.0, 1.0)));
    auto U1 = U.dup;
    const auto F = slice!double([N, N], 1.0);
    const double h2 = 1.0;

    foreach (i; 1 .. N - 1)
    {
        foreach (j; 1 .. N - 1)
        {
            if ((i + j) % 2 == Color.red)
            {
                U[i, j] = (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1] - h2 * F[i, j]) / 4.0;
            }
        }
    }

    sweep!(double, 2, Color.red)(F, U1, h2);
    assert(U == U1);

    foreach (i; 1 .. N - 1)
    {
        foreach (j; 1 .. N - 1)
        {
            if ((i + j) % 2 == Color.black)
            {
                U[i, j] = (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1] - h2 * F[i, j]) / 4.0;
            }
        }
    }

    sweep!(double, 2, Color.black)(F, U1, h2);
    assert(U == U1);

}
