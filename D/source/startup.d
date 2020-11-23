module startup;


template init()
{
    import core.thread : Thread;
    import mir.conv : to;
    import std.datetime.stopwatch : StopWatch, msecs;
    import std.experimental.logger : infof, globalLogLevel, LogLevel;
    import std.getopt : getopt;

    StopWatch sw;

    uint delay = 500;
    bool verbose = false;
    string path = default_path;
    string sweep = "ndslice";
    immutable string default_path = "../problems/problem_2D_100.npy";

    void start()
    {
        sw.reset;
        sw.start;
        globalLogLevel(LogLevel.info);

    }

    void wait_till()
    {
        if (verbose)
        {
            globalLogLevel(LogLevel.all);
        }
        auto rest = delay - sw.peek.total!"msecs";
        if (0 < rest)
        {
            Thread.sleep(msecs(rest));
        }
        sw.stop;
        sw.reset;
        sw.start;
    }

    void getopt(string[] argv)
    {
        getopt(argv, "p|P", &path, "d|D", &delay, "v", &verbose, "s", &sweep);

    }

    void print_time()
    {
        sw.stop;
        infof("%e", sw.peek
                .total!"usecs"
                .to!double / 1_000_000.0);

    }

}
