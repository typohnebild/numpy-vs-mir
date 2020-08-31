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
import multid.multigrid.restriction;

/++
This performs a GS_RB run for 3D
+/
void test3D()
{

    immutable size_t N = 50;
    auto U = slice!double(N, N, N);
    auto fun = generate!(() => uniform(0.0, 1.0));
    U.field.fill(fun);
    U[0, 0 .. $, 0 .. $] = 1.0;
    U[0 .. $, 0, 0 .. $] = 1.0;
    U[0 .. $, 0 .. $, 0] = 1.0;
    U[$ - 1, 0 .. $, 0 .. $] = 0.0;
    U[1 .. $, $ - 1, 1 .. $] = 0.0;
    U[1 .. $, 1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N, N], 0.0);
    const double h = 1.0 / double(N);

    GS_RB!(double, 3)(F, U, h);
    U.prettyArr.writeln;

}

/++
This performs a GS_RB run for 2D
+/
void test2D()
{

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

    GS_RB!(double, 2)(F, U, h);
    U.prettyArr.writeln;

}

/++
This performs a GS_RB run for 1D
+/
void test1D()
{

    immutable size_t N = 1000;
    auto U = slice!double(N);
    auto fun = generate!(() => uniform(0.0, 1.0));
    U.field.fill(fun);
    U[0] = 1.0;
    U[$ - 1] = 0.0;

    auto F = slice!double([N], 0.0);
    const double h = 1.0 / double(N);

    GS_RB!(double, 1, 30_000)(F, U, h);
    U.prettyArr.writeln;

}

void main()
{
    StopWatch sw;
    sw.reset;
    sw.start;
    test3D();

    sw.stop;
    writeln((sw.peek
            .total!"msecs"
            .to!float / 1000.0), "s");
}
