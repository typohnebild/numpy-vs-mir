module startup;

template init()
{

    import core.thread : Thread;
    import std.datetime.stopwatch : StopWatch, msecs;
    import std.experimental.logger : info, globalLogLevel, LogLevel;
    import std.getopt : getopt;
    import std.conv : to;

    StopWatch sw;

    uint delay = 500;
    bool verbose = false;
    string path = default_path;
    string sweep = "field";
    immutable string default_path = "../problems/problem_2D_100.npy";

    void start()
    {
        sw.reset;
        sw.start;
        globalLogLevel(LogLevel.info);

    }

    void wait_till()
    {
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
        if (verbose)
        {
            globalLogLevel(LogLevel.all);
        }

    }

    void print_time()
    {
        sw.stop;
        info(sw.peek
                .total!"msecs"
                .to!double / 1_000.0);

    }

}
