module multid.tools.util;

import mir.ndslice.slice: Slice;

/++ Timer Template +/
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

/++ Generator for random matrix with dimension Dim and dimension size N +/
Slice!(T*, Dim) randomMatrix(T, size_t Dim)(size_t N)
{
    import mir.random.algorithm: randomSlice;
    import mir.random.variable: uniformVar;

    size_t[Dim] shape = N;
    return uniformVar!T(0, 1).randomSlice(shape);
}
