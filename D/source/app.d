import mir.ndslice : slice;
import std.stdio : writeln;
import std.datetime.stopwatch : StopWatch, msecs;
import std.getopt : getopt;
import core.thread : Thread;
import std.conv : to;
import std.experimental.logger : info, globalLogLevel, LogLevel;

import loadproblem : npyload, getDim;
import multid.multigrid.multigrid : poisson_multigrid;
import multid.gaussseidel.redblack : GS_RB;

/++
    This loads and runs a problem that is provided on Commandline and delays the execution of
    the multigrid till delay is over.

+/
void main(string[] argv)
{
    uint delay = 500;
    StopWatch sw;
    void wait_till()
    {
        auto rest = delay - sw.peek.total!"msecs";
        if (0 < rest)
        {
            Thread.sleep(msecs(rest));
        }
        sw.stop;
        sw.reset;
    }

    globalLogLevel(LogLevel.info);

    sw.reset;
    sw.start;

    immutable string default_path = "../problems/problem_1D_100.npy";
    void warmup()
    {
        auto UF1 = npyload!(double, 1)(default_path);
        poisson_multigrid!(double, 1, 2, 2)(UF1[1].slice, UF1[0].slice, 0, 2, 1);
    }

    bool verbose = false;
    string path = default_path;
    getopt(argv, "p|P", &path, "d|D", &delay, "v", &verbose);
    if (verbose)
    {
        globalLogLevel(LogLevel.all);
    }

    const uint dim = getDim(path);

    switch (dim)
    {
    case 1:
        auto UF = npyload!(double, 1)(path);
        warmup();
        wait_till();
        sw.start;
        poisson_multigrid!(double, 1, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
        break;
    case 2:
        auto UF = npyload!(double, 2)(path);
        warmup();
        wait_till();
        sw.start;
        poisson_multigrid!(double, 2, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
        // GS_RB!(double, 2)(UF[1].slice, UF[0].slice, 1.0 / UF[0].shape[0]);
        break;
    case 3:
        //auto UF = npyload!(double, 3)(path);
        //const auto U = poisson_multigrid!(double, 3, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
        break;
    default:
        throw new Exception("wrong dimension!");
    }
    sw.stop;
    info(sw.peek
            .total!"msecs"
            .to!double / 1_000.0);
}
