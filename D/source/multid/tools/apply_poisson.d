module multid.tools.apply_poisson;

import mir.math: fastmath;
import mir.ndslice;

/++
    Calculates the A * U, where A is a poisson matrix

    Params:
        U = Dim-array
        h = distance between grid points
    Returns: x = A*U
+/
Slice!(T*, Dim) apply_poisson(T, size_t Dim)(Slice!(const(T)*, Dim) U, const T h)
{
    auto x = U.shape.slice!T;
    apply_poisson(x, U, h);
    return x;
}

@nogc @fastmath
void apply_poisson(T, size_t Dim)(Slice!(T*, Dim) x, Slice!(const(T)*, Dim) U, const T h)
{
    assumeSameShape(x, U);
    eachOnBorder!"a = b"(x, U);
    x.dropBorders[] = (1 / (h * h)) * U.withNeighboursSum.map!((u, sum) => sum - 2 * Dim * u);
}

/++
    Computes F - AU were A is the poisson matrix
+/
@nogc @fastmath
void compute_residual(T, size_t Dim)(Slice!(T*, Dim) R, Slice!(const(T)*, Dim) F, Slice!(const(T)*, Dim) U, const T current_h)
{
    assumeSameShape(U, R, F);
    // performs
    // apply_poisson(R, U, current_h);
    // R[] = F - R
    // in a single memory access
    R.dropBorders[] = ((1 / current_h ^^ 2) * U.withNeighboursSum.map!((u, sum) => sum - 2 * Dim * u)).zip!true(F.dropBorders).map!"b - a";
    eachOnBorder!"a = b - c"(R, F, U);
}

Slice!(T*, Dim) compute_residual(T, size_t Dim)(Slice!(const(T)*, Dim) F, Slice!(const(T)*, Dim) U, const T current_h)
{
    auto AU = U.shape.slice!T;
    assert(AU.shape == F.shape);
    compute_residual(AU, F, U, current_h);
    return AU;
}

unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import multid.tools.util : randomMatrix;

    const size_t N = 100;
    immutable auto h = 1.0 / double(N);

    auto U = N.randomMatrix!(double, 1);

    auto x = U.dup;
    for (size_t i = 1; i < U.shape[0] - 1; i++)
    {
        x[i] = (-2.0 * U[i] + U[i - 1] + U[i + 1]) / (h * h);
    }

    auto x1 = apply_poisson(U, h);
    assert(all!approxEqual(x, x1));
}

unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import multid.tools.util : randomMatrix;

    const size_t N = 100;
    immutable auto h = 1.0 / double(N);

    auto U = N.randomMatrix!(double, 2);

    immutable m = U.shape[0];
    immutable n = U.shape[1];
    auto x = U.dup;

    for (size_t i = 1; i < m - 1; i++)
    {
        for (size_t j = 1; j < n - 1; j++)
        {
            x[i, j] = (-4.0 * U[i, j]
                + U[i - 1, j]
                + U[i + 1, j]
                + U[i, j - 1]
                + U[i, j + 1]) / (h * h);
        }
    }

    auto x1 = apply_poisson(U, h);
    assert(all!approxEqual(x, x1));
}


unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import multid.tools.util : randomMatrix;

    const size_t N = 100;
    immutable auto h = 1.0 / double(N);

    auto U = N.randomMatrix!(double, 3);

    auto x = U.dup;
    for (size_t i = 1; i < U.shape[0] - 1; i++)
    {
        for (size_t j = 1; j < U.shape[1] - 1; j++)
        {
            for (size_t k = 1; k < U.shape[2] - 1; k++)
            {
                x[i, j, k] = (-6.0 *
                        U[i, j, k] +
                        U[i - 1, j, k] +
                        U[i + 1, j, k] +
                        U[i, j - 1, k] +
                        U[i, j + 1, k] +
                        U[i, j, k - 1] +
                        U[i, j, k + 1]) / (h * h);
            }
        }
    }

    auto x1 = apply_poisson(U, h);
    assert(all!approxEqual(x, x1));
}
