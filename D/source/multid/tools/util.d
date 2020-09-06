module multid.tools.util;

template Timer()
{
    import std.datetime.stopwatch;
    import std.stdio : writeln;

    StopWatch sw;

    void start()
    {
        sw.reset;
        sw.start;
    }

    void stop(string text)
    {
        sw.stop;
        writeln(text, " ", sw.peek.total!"msecs");
    }
}
