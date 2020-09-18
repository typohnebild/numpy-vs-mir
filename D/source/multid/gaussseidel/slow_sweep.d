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
