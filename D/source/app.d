import std.range, std.stdio;

void main()
{
    auto sum = 30.0;
    auto count = 50;//stdin.byLine.tee!(l => sum += l.length).walkLength;

    write("Average line length: ", count ? sum / count : 0);
}