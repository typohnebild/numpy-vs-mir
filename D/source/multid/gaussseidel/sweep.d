module multid.gaussseidel.sweep;

import mir.math: fastmath;
import mir.algorithm.iteration: Chequer, each;
import mir.ndslice : assumeSameShape, slice, sliced, Slice, SliceKind, strided, dropBorders, withNeighboursSum, zip;


@nogc @fastmath
void sweep_ndslice(T, size_t N)(Chequer color, Slice!(const(T)*, N) F, Slice!(T*, N) U, const T h2) nothrow
{
    // find the naive implementation R/B order
    color = N % 2 ? cast(Chequer)!color : color;
    assumeSameShape(F, U);
    color.each!(z => z.a.a = (T(1) / (2 * N)) * (z.a.b - h2 * z.b))
        (U.withNeighboursSum.zip!true(F.dropBorders));
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
@nogc @fastmath
void sweep_field(T)(Chequer color, Slice!(const(T)*, 1) F, Slice!(T*, 1) U, const T h2) nothrow
{
    const N = F.shape[0];
    auto UF = U.field;
    auto FF = F.field;
    for (size_t i = 2u - color; i < N - 1u; i += 2u)
    {
        UF[i] = (UF[i - 1u] + UF[i + 1u] - FF[i] * h2) / 2;
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
@nogc @fastmath
void sweep_field(T)(Chequer color, Slice!(const(T)*, 2) F, Slice!(T*, 2) U, const T h2) nothrow
{
    const m = F.shape[0];
    const n = F.shape[1];
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
                    UF[flatindex + 1] - h2 * FF[flatindex]) / 4;
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
@nogc @fastmath
void sweep_field(T)(Chequer color, Slice!(const(T)*, 3) F, Slice!(T*, 3) U, const T h2) nothrow
{
    const m = F.shape[0];
    const n = F.shape[1];
    const l = F.shape[2];
    auto UF = U.field;
    auto FF = F.field;
    foreach (i; 1 .. m - 1)
    {
        foreach (j; 1 .. n - 1)
        {
            const flatindex2d = i * (n * l) + j * l;
            for (size_t k = 1u + (i + j + 1 + color) % 2; k < l - 1u; k += 2)
            {
                const flatindex = flatindex2d + k;
                UF[flatindex] = (
                        UF[flatindex - n * l] +
                        UF[flatindex + n * l] +
                        UF[flatindex - l] +
                        UF[flatindex + l] +
                        UF[flatindex - 1] +
                        UF[flatindex + 1] - h2 * FF[flatindex]) * (T(1) / 6);
            }
        }
    }
}

private struct SweepKernel(T, size_t Dim)
{
    import std.meta: Repeat;

    T h2;

    this(T h2)
    {
        this.h2 = h2;
    }

    @fastmath
    void opCall()(ref scope T r, ref scope const Repeat!(2 * Dim, T) neighbors, ref scope const T f) const
    {
        T sum = neighbors[0];
        foreach (ref neighbor; neighbors[1 .. $])
            sum += neighbor;
        r = (sum - f * h2) * (T(1) / neighbors.length);
    }
}

/++ slow sweep for 1D +/
@nogc @fastmath
void sweep_slice(T)(Chequer color, Slice!(const(T)*, 1) F, Slice!(T*, 1) U, const T h2) nothrow
{
    assumeSameShape(F, U);
    auto kernel = SweepKernel!(T, 1)(h2);

    each!kernel(
        U[2 - color .. $ - 1].strided(2),
        U[1 - color .. $ - 2].strided(2),
        U[3 - color .. $].strided(2),
        F[2 - color .. $ - 1].strided(2));
}

/++ slow sweep for 2D +/
@nogc @fastmath
void sweep_slice(T)(Chequer color, Slice!(const(T)*, 2) F, Slice!(T*, 2) U, const T h2) nothrow
{
    assumeSameShape(F, U);
    auto kernel = SweepKernel!(T, 2)(h2);

    each!kernel(
        U[1 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[0 .. $ - 2, 1 + color .. $ - 1].strided(2),
        U[2 .. $, 1 + color .. $ - 1].strided(2),
        U[1 .. $ - 1, color .. $ - 2].strided(2),
        U[1 .. $ - 1, 2 + color .. $].strided(2),
        F[1 .. $ - 1, 1 + color .. $ - 1].strided(2));

    each!kernel(
        U[2 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[1 .. $ - 2, 2 - color .. $ - 1].strided(2),
        U[3 .. $, 2 - color .. $ - 1].strided(2),
        U[2 .. $ - 1, 1 - color .. $ - 2].strided(2),
        U[2 .. $ - 1, 3 - color .. $].strided(2),
        F[2 .. $ - 1, 2 - color .. $ - 1].strided(2));
}

/++ slow sweep for 3D +/
@nogc @fastmath
void sweep_slice(T)(Chequer color, Slice!(const(T)*, 3) F, Slice!(T*, 3) U, const T h2) nothrow
{
    assumeSameShape(F, U);
    auto kernel = SweepKernel!(T, 3)(h2);

    each!kernel(
        U[2 .. $ - 1, 1 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[1 .. $ - 2, 1 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[3 .. $, 1 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[2 .. $ - 1, 0 .. $ - 2, 1 + color .. $ - 1].strided(2),
        U[2 .. $ - 1, 2 .. $, 1 + color .. $ - 1].strided(2),
        U[2 .. $ - 1, 1 .. $ - 1, color .. $ - 2].strided(2),
        U[2 .. $ - 1, 1 .. $ - 1, 2 + color .. $].strided(2),
        F[2 .. $ - 1, 1 .. $ - 1, 1 + color .. $ - 1].strided(2));

    each!kernel(
        U[1 .. $ - 1, 1 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[0 .. $ - 2, 1 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[2 .. $, 1 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[1 .. $ - 1, 0 .. $ - 2, 2 - color .. $ - 1].strided(2),
        U[1 .. $ - 1, 2 .. $, 2 - color .. $ - 1].strided(2),
        U[1 .. $ - 1, 1 .. $ - 1, 1 - color .. $ - 2].strided(2),
        U[1 .. $ - 1, 1 .. $ - 1, 3 - color .. $].strided(2),
        F[1 .. $ - 1, 1 .. $ - 1, 2 - color .. $ - 1].strided(2));

    each!kernel(
        U[1 .. $ - 1, 2 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[0 .. $ - 2, 2 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[2 .. $, 2 .. $ - 1, 1 + color .. $ - 1].strided(2),
        U[1 .. $ - 1, 1 .. $ - 2, 1 + color .. $ - 1].strided(2),
        U[1 .. $ - 1, 3 .. $, 1 + color .. $ - 1].strided(2),
        U[1 .. $ - 1, 2 .. $ - 1, color .. $ - 2].strided(2),
        U[1 .. $ - 1, 2 .. $ - 1, 2 + color .. $].strided(2),
        F[1 .. $ - 1, 2 .. $ - 1, 1 + color .. $ - 1].strided(2));

    each!kernel(
        U[2 .. $ - 1, 2 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[1 .. $ - 2, 2 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[3 .. $, 2 .. $ - 1, 2 - color .. $ - 1].strided(2),
        U[2 .. $ - 1, 1 .. $ - 2, 2 - color .. $ - 1].strided(2),
        U[2 .. $ - 1, 3 .. $, 2 - color .. $ - 1].strided(2),
        U[2 .. $ - 1, 2 .. $ - 1, 1 - color .. $ - 2].strided(2),
        U[2 .. $ - 1, 2 .. $ - 1, 3 - color .. $].strided(2),
        F[2 .. $ - 1, 2 .. $ - 1, 2 - color .. $ - 1].strided(2));
}

/++ naive sweep for 1D +/
@nogc @fastmath
void sweep_naive(T)(Chequer color, Slice!(const(T)*, 1) F, Slice!(T*, 1) U, const T h2) nothrow
{
    foreach (i; 1 .. F.length - 1)
        if (i % 2 == color)
            U[i] = (U[i - 1u] + U[i + 1u] - F[i] * h2) / 2;
}
/++ naive sweep for 2D +/
@nogc @fastmath
void sweep_naive(T)(Chequer color, Slice!(const(T)*, 2) F, Slice!(T*, 2) U, const T h2) nothrow
{
    foreach (i; 1 .. F.length!0 - 1)
    foreach (j; 1 .. F.length!1 - 1)
        if ((i + j) % 2 == color)
            U[i, j] = (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1] - h2 * F[i, j]) / 4;
}
/++ naive sweep for 3D +/
@nogc @fastmath
void sweep_naive(T)(Chequer color, Slice!(const(T)*, 3) F, Slice!(T*, 3) U, const T h2) nothrow
{
    foreach (i; 1 .. F.length!0 - 1)
    foreach (j; 1 .. F.length!1 - 1)
    foreach (k; 1 .. F.length!2 - 1)
        if ((i + j + k) % 2 == color)
            U[i, j, k] = (T(1) / 6) *
                ( U[i - 1, j, k]
                + U[i + 1, j, k]
                + U[i, j - 1, k]
                + U[i, j + 1, k]
                + U[i, j, k - 1]
                + U[i, j, k + 1]
                - h2 * F[i, j, k]);
}
