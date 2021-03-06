import mir.ndslice : slice;
import std.exception : enforce;

import startup : init;
import loadproblem : npyload, getDim;
import multid.gaussseidel.redblack : GS_RB, SweepType;

/++
    This loads and runs a problem that is provided on Commandline and delays the execution of
    the Gauss-Seidel redblack till delay is over.

+/
void main(string[] argv)
{
    alias i = init!();
    i.start();
    i.getopt(argv);
    const iterations = 5_000;

    void warmup()
    {
        auto UF1 = npyload!(double, 2)(i.default_path);
        GS_RB!(SweepType.ndslice)(UF1[1].slice, UF1[0].slice, 1, iterations, iterations + 10, 1e-8);
    }

    const uint dim = getDim(i.path);
    enforce(dim == 2, "This benchmark only supports 2D problems");

    auto UF = npyload!(double, 2)(i.path);
    warmup();
    i.wait_till();
    switch (i.sweep)
    {
    case "slice":
        GS_RB!(SweepType.slice)(UF[1].slice, UF[0].slice, 1, iterations, iterations + 10, 1e-8);
        break;
    case "naive":
        GS_RB!(SweepType.naive)(UF[1].slice, UF[0].slice, 1, iterations, iterations + 10, 1e-8);
        break;
    case "field":
        GS_RB!(SweepType.field)(UF[1].slice, UF[0].slice, 1, iterations, iterations + 10, 1e-8);
        break;
    default:
        GS_RB!(SweepType.ndslice)(UF[1].slice, UF[0].slice, 1, iterations, iterations + 10, 1e-8);

    }
    i.print_time();
}
