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

    void warmup()
    {
        auto UF1 = npyload!(double, 2)(i.default_path);
        GS_RB!(double, 2, 10_000_000, 1_000, 1e-8, SweepType.field)(UF1[1].slice, UF1[0].slice, 1);
    }

    const uint dim = getDim(i.path);
    enforce(dim == 2, "This benchmark only supports 2D problems");

    auto UF = npyload!(double, 2)(i.path);
    warmup();
    i.wait_till();
    switch (i.sweep)
    {
    case "slice":
        GS_RB!(double, 2, 10_000_000, 1_000, 1e-8, SweepType.slice)(UF[1].slice, UF[0].slice, 1);
        break;
    case "naive":
        GS_RB!(double, 2, 10_000_000, 1_000, 1e-8, SweepType.naive)(UF[1].slice, UF[0].slice, 1);
        break;
    default:
        GS_RB!(double, 2, 10_000_000, 1_000, 1e-8, SweepType.field)(UF[1].slice, UF[0].slice, 1);

    }
    i.print_time();
}
