import std.stdio;
import std.range : generate;
import std.random : uniform;
import std.array;
import std.algorithm;
import mir.ndslice;
import pretty_array;
import std.datetime.stopwatch : StopWatch;
import std.conv : to;

import multid.gaussseidel.redblack;

void main()
{
    StopWatch sw;
    sw.reset;
    sw.start;

    immutable size_t N = 100;
    auto U = slice!double(N, N);
    auto fun = generate!(() => uniform(0.0, 1.0));
    U.field.fill(fun);
    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    const double h = 1.0 / double(N);

    GS_RB!(double, 2, 30_000)(F, U, h);
    sw.stop;
    writeln((sw.peek
            .total!"msecs"
            .to!float / 1000.0), "s");
    U.prettyArr.writeln;
}
