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
import multid.multigrid.cycle;
import multid.multigrid.multigrid;
import loadproblem;

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

    immutable size_t N = 200;
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

/++
This performs multigrid for 2D
+/
void testMG2D()
{
    immutable size_t N = 1000;
    auto U = slice!double(N, N);
    auto fun = generate!(() => uniform(0.0, 1.0));
    U.field.fill(fun);
    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    F[0][0 .. $] = 1.0;
    F[1 .. $, 0] = 1.0;
    F[$ - 1][1 .. $] = 0.0;
    F[1 .. $, $ - 1] = 0.0;

    U = poisson_multigrid!(double, 2, 2, 2)(F, U, 0, 2, 100);

    //U.prettyArr.writeln;
}

void main(string[] argv)
{
    StopWatch sw;
    sw.reset;
    sw.start;
    //testMG2D();


    //string pfad = argv[1]; //"../problems/problem_2D_100.npy";
    string pfad = "../problems/problem_1D_100.npy";
    const uint dim = getDim(pfad);

    switch (dim)
    {
        case 1:
            auto UF = npyload!(double, 1)(pfad);
            const auto U = poisson_multigrid!(double, 1, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
            U.prettyArr.writeln;
            break;
        case 2:
            auto UF = npyload!(double, 2)(pfad);
            const auto U = poisson_multigrid!(double, 2, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
            U.prettyArr.writeln;
            break;
        case 3:
            //auto UF = npyload!(double, 3)(pfad);
            //const auto U = poisson_multigrid!(double, 3, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
            //U.prettyArr.writeln;
            break;
        default:
            throw new Exception("wrong dimension!");
    }


    sw.stop;
    writeln((sw.peek
            .total!"msecs"
            .to!float / 1000.0), "s");
}
