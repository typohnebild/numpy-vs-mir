module multid.tools.apply_poisson;

import mir.ndslice;
import std.stdio : writeln;

/++
    Calculates the A * U, where A is a poisson matrix

    Params:
        U = Dim-array
        h = distance between grid points
    Returns: x = A*U
+/
Slice!(T*, Dim) apply_poisson(T, size_t Dim)(Slice!(T*, Dim) U, T h)
{
    auto x = U.dup;
    const T h2 = h * h;

    static if (Dim == 1)
    {
        for (size_t i = 1; i < U.shape[0] - 1; i++)
        {
            x[i] = (-2.0 * U[i] + U[i - 1] + U[i + 1]) / h2;
        }

    }
    else static if (Dim == 2)
    {
        for (size_t i = 1; i < U.shape[0] - 1; i++)
        {
            for (size_t j = 1; j < U.shape[1] - 1; j++)
            {
                x[i, j] = (-4.0 * U[i, j] + U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1]) / h2;

            }
        }

    }
    else static if (Dim == 3)
    {
        for (size_t i = 1; i < U.shape[0] - 1; i++)
        {
            for (size_t j = 1; j < U.shape[1] - 1; j++)
            {
                for (size_t k = 1; k < U.shape[2] - 1; k++)
                {
                    x[i, j, k] = (-6.0 * U[i, j, k] + U[i - 1, j, k] + U[i + 1,
                            j, k] + U[i, j - 1, k] + U[i, j + 1, k] + U[i, j, k - 1] + U[i, j, k + 1]) / h2;

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
