module multid.tools.util;
import mir.ndslice : Slice, slice;
import std.range : generate;
import std.random : uniform;
import std.algorithm : fill;

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

Slice!(T*, Dim) randomMatrix(T, size_t Dim)(size_t N)
{
    size_t[Dim] shape = N;
    auto ret = slice!T(shape);
    ret.field.fill(generate!(() => uniform(0.0, 1.0)));
    return ret;
}
