import mir.ndslice : slice;
import std.stdio : writeln;
import std.datetime.stopwatch : StopWatch, msecs;
import std.getopt : getopt;
import core.thread : Thread;
import std.conv : to;

import loadproblem;
import multid.multigrid.multigrid;

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

    sw.reset;
    sw.start;

    string path = "../problems/problem_1D_100.npy";
    getopt(argv, "p|P", &path, "d|D", &delay);

    const uint dim = getDim(path);

    switch (dim)
    {
    case 1:
        auto UF = npyload!(double, 1)(path);
        wait_till();
        sw.start;
        poisson_multigrid!(double, 1, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
        break;
    case 2:
        auto UF = npyload!(double, 2)(path);
        wait_till();
        sw.start;
        poisson_multigrid!(double, 2, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
        break;
    case 3:
        //auto UF = npyload!(double, 3)(path);
        //const auto U = poisson_multigrid!(double, 3, 2, 2)(UF[1].slice, UF[0].slice, 0, 2, 100);
        break;
    default:
        throw new Exception("wrong dimension!");
    }
    sw.stop;
    writeln(sw.peek
            .total!"msecs"
            .to!double / 1_000.0);
}
