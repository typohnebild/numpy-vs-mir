import std.stdio;
import std.range : generate;
import std.random : uniform;
import std.array;
import std.algorithm;
import mir.ndslice;
import pretty_array;

import multid.gaussseidel.redblack;

void main()
{
    const size_t N = 100;
    auto U = slice!double(N, N);
    auto fun = generate!(() => uniform(0.0, 1.0));
    U.field.fill(fun);
    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;
    U.prettyArr.writeln;

    auto F = slice!double([N, N], 0.0);
    const double h = 1 / N;
    GS_RB!(double, 2, 666)(F, U, h);
    U.prettyArr.writeln;
}
