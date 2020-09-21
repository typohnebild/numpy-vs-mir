module multid.gaussseidel.slow_sweep;

import mir.ndslice : slice, sliced, Slice, strided;
import multid.gaussseidel.redblack : Color;

/++ slow sweep for 1D +/
void slow_sweep(T, size_t Dim : 1, Color color)(in Slice!(T*, 1) F, Slice!(T*, 1) U, in T h2)
{
    U[2 - color .. $ - 1].strided!0(2)[] = (
            U[1 - color .. $ - 2].strided!0(2) + U[3 - color .. $].strided!0(
            2) - h2 * F[2 - color .. $ - 1].strided!0(2)) / 2.0;
}

/++ slow sweep for 2D +/
void slow_sweep(T, size_t Dim : 2, Color color)(in Slice!(T*, 2) F, Slice!(T*, 2) U, in T h2)
{
    const auto m = F.shape[0];
    const auto n = F.shape[1];
    auto strideU = U[1 .. m - 1, 1 + color .. n - 1].strided!(0, 1)(2, 2);
    strideU[] = U[0 .. m - 2, 1 + color .. n - 1].strided!(0, 1)(2, 2);
    strideU[] += U[2 .. m, 1 + color .. n - 1].strided!(0, 1)(2, 2);
    strideU[] += U[1 .. m - 1, color .. n - 2].strided!(0, 1)(2, 2);
    strideU[] += U[1 .. m - 1, 2 + color .. n].strided!(0, 1)(2, 2);
    strideU[] -= F[1 .. m - 1, 1 + color .. n - 1].strided!(0, 1)(2, 2) * h2;
    strideU[] /= cast(T) 4;

    strideU = U[2 .. m - 1, 2 - color .. n - 1].strided!(0, 1)(2, 2);
    strideU[] = U[1 .. m - 2, 2 - color .. n - 1].strided!(0, 1)(2, 2);
    strideU[] += U[3 .. m, 2 - color .. n - 1].strided!(0, 1)(2, 2);
    strideU[] += U[2 .. m - 1, 1 - color .. n - 2].strided!(0, 1)(2, 2);
    strideU[] += U[2 .. m - 1, 3 - color .. n].strided!(0, 1)(2, 2);
    strideU[] -= F[2 .. m - 1, 2 - color .. n - 1].strided!(0, 1)(2, 2) * h2;
    strideU[] /= cast(T) 4;
}
/++ slow sweep for 3D +/
void slow_sweep(T, size_t Dim : 3, Color color)(in Slice!(T*, 3) F, Slice!(T*, 3) U, in T h2)
{
    const auto m = F.shape[0];
    const auto n = F.shape[1];
    const auto o = F.shape[2];

    auto strideU = U[2 .. m - 1, 1 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] = U[1 .. m - 2, 1 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[3 .. m, 1 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 0 .. n - 2, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 2 .. n, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 1 .. n - 1, color .. o - 2].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 1 .. n - 1, 2 + color .. o].strided!(0, 1, 2)(2, 2, 2);
    strideU[] -= F[2 .. m - 1, 1 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2) * h2;
    strideU[] /= cast(T) 6;

    strideU = U[1 .. m - 1, 1 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] = U[0 .. m - 2, 1 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m, 1 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 0 .. n - 2, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 2 .. n, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 1 .. n - 1, 1 - color .. o - 2].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 1 .. n - 1, 3 - color .. o].strided!(0, 1, 2)(2, 2, 2);
    strideU[] -= F[1 .. m - 1, 1 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2) * h2;
    strideU[] /= cast(T) 6;

    strideU = U[1 .. m - 1, 2 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] = U[0 .. m - 2, 2 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m, 2 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 1 .. n - 2, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 3 .. n, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 2 .. n - 1, color .. o - 2].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[1 .. m - 1, 2 .. n - 1, 2 + color .. o].strided!(0, 1, 2)(2, 2, 2);
    strideU[] -= F[1 .. m - 1, 2 .. n - 1, 1 + color .. o - 1].strided!(0, 1, 2)(2, 2, 2) * h2;
    strideU[] /= cast(T) 6;

    strideU = U[2 .. m - 1, 2 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] = U[1 .. m - 2, 2 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[3 .. m, 2 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 1 .. n - 2, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 3 .. n, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 2 .. n - 1, 1 - color .. o - 2].strided!(0, 1, 2)(2, 2, 2);
    strideU[] += U[2 .. m - 1, 2 .. n - 1, 3 - color .. o].strided!(0, 1, 2)(2, 2, 2);
    strideU[] -= F[2 .. m - 1, 2 .. n - 1, 2 - color .. o - 1].strided!(0, 1, 2)(2, 2, 2) * h2;
    strideU[] /= cast(T) 6;

}

void sweep_naive(T, size_t Dim : 1, Color color)(const Slice!(T*, 1) F, Slice!(T*, 1) U, T h2)
{

    const auto n = F.shape[0];
    foreach (i; 1 .. n - 1)
    {
        if (i % 2 == color)
        {
            U[i] = (U[i - 1u] + U[i + 1u] - F[i] * h2) / 2.0;
        }
    }

}

void sweep_naive(T, size_t Dim : 2, Color color)(const Slice!(T*, 2) F, Slice!(T*, 2) U, T h2)
{
    const auto n = F.shape[0];
    const auto m = F.shape[1];

    foreach (i; 1 .. n - 1)
    {
        foreach (j; 1 .. m - 1)
        {
            if ((i + j) % 2 == color)
            {
                U[i, j] = (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1] - h2 * F[i, j]) / 4.0;
            }
        }
    }
}

void sweep_naive(T, size_t Dim : 3, Color color)(const Slice!(T*, 3) F, Slice!(T*, 3) U, T h2)
{
    const auto n = F.shape[0];
    const auto m = F.shape[1];
    const auto l = F.shape[2];
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
