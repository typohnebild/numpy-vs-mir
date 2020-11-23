import mir.ndslice : slice;

import startup : init;
import loadproblem : npyload, getDim;
import multid.multigrid.multigrid : poisson_multigrid;
import multid.gaussseidel.redblack : GS_RB;

/++
    This loads and runs a problem that is provided on Commandline and delays the execution of
    the multigrid till delay is over.

+/
void main(string[] argv)
{
    alias i = init!();
    i.start();
    i.getopt(argv);

    void warmup()
    {
        auto UF1 = npyload!(double, 2)(i.default_path);
        poisson_multigrid(UF1[1].slice, UF1[0].slice, 0, 1, 2, 2, 1);
    }

    const uint dim = getDim(i.path);

    switch (dim)
    {
    case 1:
        auto UF = npyload!(double, 1)(i.path);
        warmup();
        i.wait_till();
        poisson_multigrid(UF[1].slice, UF[0].slice, 0, 1, 2, 2, 100, i.sweep);
        break;
    case 2:
        auto UF = npyload!(double, 2)(i.path);
        warmup();
        i.wait_till();
        poisson_multigrid(UF[1].slice, UF[0].slice, 0, 1, 2, 2, 100, i.sweep);
        break;
    case 3:
        auto UF = npyload!(double, 3)(i.path);
        warmup();
        i.wait_till();
        poisson_multigrid(UF[1].slice, UF[0].slice, 0, 1, 2, 2, 100, i.sweep);
        break;
    default:
        throw new Exception("wrong dimension!");
    }
    i.print_time();
}
