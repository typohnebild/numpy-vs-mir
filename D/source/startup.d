module startup;


template init()
{

    import core.thread : Thread;
    import core.time : convert;
    import std.datetime.stopwatch : StopWatch, hnsecs;
    import std.datetime.systime : Clock, stdTimeToUnixTime, unixTimeToStdTime;
    import std.experimental.logger : infof, globalLogLevel, LogLevel;
    import std.getopt : getopt;
    import std.conv : to;
    import std.exception : enforce;

    StopWatch sw;

    int starttime;
    int delay = 500;
    bool verbose = false;
    string path = default_path;
    string sweep = "field";
    immutable string default_path = "../problems/problem_2D_100.npy";

    void start()
    {
        globalLogLevel(LogLevel.info);
    }

    void wait_till()
    {
        if (verbose)
        {
            globalLogLevel(LogLevel.all);
        }
        auto x = (Clock.currStdTime - starttime.unixTimeToStdTime).convert!("hnsecs", "msecs");
        auto rest = delay.convert!("msecs", "hnsecs") - x;

        enforce(0 < rest, "Warmup took to long");

        Thread.sleep(hnsecs(rest));
        sw.stop;
        sw.reset;
        sw.start;
    }

    void getopt(string[] argv)
    {
        getopt(argv, "p|P", &path, "d|D", &delay, "v", &verbose, "s", &sweep, "t", &starttime);

    }

    void print_time()
    {
        sw.stop;
        infof("%e", sw.peek
                .total!"usecs"
                .to!double / 1_000_000.0);

    }

}
