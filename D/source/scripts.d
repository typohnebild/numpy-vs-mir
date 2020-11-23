import loadproblem;
import mir.ndslice;
import multid.gaussseidel.redblack;
import multid.multigrid.cycle;
import multid.multigrid.multigrid;
import multid.multigrid.restriction;
import multid.tools.util;
// import pretty_array;
import std.datetime.stopwatch : StopWatch;
import std.stdio;

/++
This performs a GS_RB run for 3D
+/
void test3D()
{

    immutable size_t N = 50;
    auto U = N.randomMatrix!(double, 3);
    U[0, 0 .. $, 0 .. $] = 1.0;
    U[0 .. $, 0, 0 .. $] = 1.0;
    U[0 .. $, 0 .. $, 0] = 1.0;
    U[$ - 1, 0 .. $, 0 .. $] = 0.0;
    U[1 .. $, $ - 1, 1 .. $] = 0.0;
    U[1 .. $, 1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N, N], 0.0);
    const double h = 1.0 / double(N);

    GS_RB(F, U, h);
    // U.prettyArr.writeln;

}

/++
This performs a GS_RB run for 2D
+/
void test2D()
{

    immutable size_t N = 200;
    auto U = N.randomMatrix!(double, 2);
    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    const double h = 1.0 / double(N);

    GS_RB(F, U, h);
    // U.prettyArr.writeln;

}

/++
This performs a GS_RB run for 1D
+/
void test1D()
{

    immutable size_t N = 1000;
    auto U = N.randomMatrix!(double, 1);
    U[0] = 1.0;
    U[$ - 1] = 0.0;

    auto F = slice!double([N], 0.0);
    const double h = 1.0 / double(N);

    GS_RB(F, U, h, 30_000);
    // U.prettyArr.writeln;

}

/++
This performs multigrid for 2D
+/
void testMG2D()
{
    immutable size_t N = 1000;
    auto U = N.randomMatrix!(double, 2);
    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    F[0][0 .. $] = 1.0;
    F[1 .. $, 0] = 1.0;
    F[$ - 1][1 .. $] = 0.0;
    F[1 .. $, $ - 1] = 0.0;

    U = poisson_multigrid(F, U, 0, 2, 2, 2, 100);

    //U.prettyArr.writeln;
}
