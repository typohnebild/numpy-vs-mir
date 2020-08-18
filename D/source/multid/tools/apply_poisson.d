module multid.tools.apply_poisson;

import mir.ndslice;
import std.stdio : writeln;

/++
    Calculates the A * U, where A is a poisson matrix
+/
Slice!(T*, Dim) apply_poisson(T, size_t Dim : 1)(Slice!(T*, Dim) U, T h)
{
    auto x = U.dup;
    x[0] = U[0];
    x[$ - 1] = U[$ - 1];
    for (size_t i = 1; i < U.shape[0] - 1; i++)
    {
        x[i] = (-2.0 * U[i] + U[i - 1] + U[i + 1]) / (h * h);
    }
    return x;
}

/++
    Calculates the A * U, where A is a poisson matrix
+/
Slice!(T*, Dim) apply_poisson(T, size_t Dim : 2)(Slice!(T*, Dim) U, T h)
{
    auto x = U.dup;
    for (size_t i = 1; i < U.shape[0] - 1; i++)
    {
        for (size_t j = 1; j < U.shape[1] - 1; j++)
        {
            x[i, j] = (-4.0 * U[i, j] + U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1]) / (
                    h * h);

        }
    }
    return x;
}
