module multid.gaussseidel.slow_sweep;

import mir.ndslice : slice, sliced, Slice, strided;
import multid.gaussseidel.redblack : Color;

void slow_sweep(T, size_t Dim : 2, Color color)(in Slice!(T*, 2) F, Slice!(T*, 2) U, in T h2)
{
    const auto m = F.shape[0];
    const auto n = F.shape[1];
    static if (color == Color.red)
    {
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] = U[0 .. m - 2, 1 .. n - 1].strided!(0, 1)(2, 2);
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m, 1 .. n - 1].strided!(0, 1)(2, 2);
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[1 .. m - 1, 0 .. n - 2].strided!(0, 1)(2, 2);
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] += U[1 .. m - 1, 2 .. n].strided!(0, 1)(2, 2);
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] -= F[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2) * h2;
        U[1 .. m - 1, 1 .. n - 1].strided!(0, 1)(2, 2)[] /= cast(T) 4.0;

        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] = U[1 .. m - 2, 2 .. n - 1].strided!(0, 1)(2, 2);
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[3 .. m, 2 .. n - 1].strided!(0, 1)(2, 2);
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m - 1, 1 .. n - 2].strided!(0, 1)(2, 2);
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] += U[2 .. m - 1, 3 .. n].strided!(0, 1)(2, 2);
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] -= F[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2) * h2;
        U[2 .. m - 1, 2 .. n - 1].strided!(0, 1)(2, 2)[] /= cast(T) 4.0;
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
