module multid.tools.apply_poisson;

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
    auto x = slice!(T)(U.shape);
    const T h2 = h * h;
    auto UF = U.field;

    static if (Dim == 1)
    {
        x.field[0] = UF[0];
        x.field[$ - 1] = UF[$ - 1];
        foreach (i; 1 .. U.shape[0] - 1)
        {
            x.field[i] = (-2.0 * UF[i] + UF[i - 1] + UF[i + 1]) / h2;
        }

    }
    else static if (Dim == 2)
    {
        immutable m = U.shape[0];
        immutable n = U.shape[1];

        x.field[0 .. m] = UF[0 .. m];
        x.field[$ - m .. $] = UF[$ - m .. $];

        x[1 .. $ - 1, 0] = U[1 .. $ - 1, 0];
        x[1 .. $ - 1, $ - 1] = U[1 .. $ - 1, $ - 1];

        foreach (i; 1 .. m - 1)
        {
            foreach (j; 1 .. n - 1)
            {
                auto flatindex = i * m + j;
                x.field[flatindex] = (
                        -4.0 * UF[flatindex] +
                        UF[flatindex - m] +
                        UF[flatindex + m] +
                        UF[flatindex - 1] +
                        UF[flatindex + 1]) / h2;
            }
        }
    }
    else static if (Dim == 3)
    {

        const auto m = U.shape[0];
        const auto n = U.shape[1];
        const auto l = U.shape[2];

        x[0 .. $, 0 .. $, 0] = U[0 .. $, 0 .. $, 0];
        x[0 .. $, 0, 0 .. $] = U[0 .. $, 0, 0 .. $];
        x[0, 0 .. $, 0 .. $] = U[0, 0 .. $, 0 .. $];

        x[0 .. $, 0 .. $, $ - 1] = U[0 .. $, 0 .. $, $ - 1];
        x[0 .. $, $ - 1, 0 .. $] = U[0 .. $, $ - 1, 0 .. $];
        x[$ - 1, 0 .. $, 0 .. $] = U[$ - 1, 0 .. $, 0 .. $];

        for (size_t i = 1; i < m - 1; i++)
        {
            for (size_t j = 1; j < n - 1; j++)
            {
                const auto flatindex2d = i * (n * l) + j * l;
                for (size_t k = 1; k < l - 1; k++)
                {
                    const flatindex = flatindex2d + k;
                    x.field[flatindex] = (
                            -6.0 *
                            UF[flatindex] +
                            UF[flatindex - n * l] +
                            UF[flatindex + n * l] +
                            UF[flatindex - l] +
                            UF[flatindex + l] +
                            UF[flatindex - 1] +
                            UF[flatindex + 1]) / h2;
                }
            }
        }
    }
    else
    {
        static assert(false, Dim.stringof ~ " is not a supported dimension!");
    }

    return x;
}

/++
    Computes F - AU were A is the poisson matrix
+/
Slice!(T*, Dim) compute_residual(T, size_t Dim)(Slice!(const(T)*, Dim) F, Slice!(const(T)*, Dim) U, const T current_h)
{
    auto AU = apply_poisson!(T, Dim)(U, current_h);
    AU.field[] = F.field[] - AU.field[];
    return AU;
}

unittest
{
    import multid.tools.util : randomMatrix;

    const size_t N = 100;
    immutable auto h = 1.0 / double(N);

    auto U = randomMatrix!(double, 1)(N);

    auto x = U.dup;
    for (size_t i = 1; i < U.shape[0] - 1; i++)
    {
        x[i] = (-2.0 * U[i] + U[i - 1] + U[i + 1]) / (h * h);
    }

    const auto x1 = apply_poisson!(double, 1)(U, h);
    assert(x == x1);
}

unittest
{

    import multid.tools.util : randomMatrix;

    const size_t N = 100;
    immutable auto h = 1.0 / double(N);

    auto U = randomMatrix!(double, 2)(N);

    immutable m = U.shape[0];
    immutable n = U.shape[1];
    auto x = U.dup;

    for (size_t i = 1; i < m - 1; i++)
    {
        for (size_t j = 1; j < n - 1; j++)
        {
            x[i, j] = (-4.0 * U[i, j] + U[i - 1, j] + U[i + 1, j] + U[i, j - 1]
                    + U[i, j + 1]) / (h * h);
        }
    }
    const auto x1 = apply_poisson!(double, 2)(U, h);
    assert(x == x1);
}


unittest
{
    import multid.tools.util : randomMatrix;

    const size_t N = 100;
    immutable auto h = 1.0 / double(N);

    auto U = randomMatrix!(double, 3)(N);

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

    const auto x1 = apply_poisson!(double, 3)(U, h);

    assert(x == x1);
}
