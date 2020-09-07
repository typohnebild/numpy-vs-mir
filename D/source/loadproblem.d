module loadproblem;

import mir.ndslice;
import numir.io;
import std.regex;
import std.stdio;
import std.conv: to;


// /++
// Implementation of npy loader
// +/
// auto npyloader(string path)
// {
//     uint dim = getDim(path);
//     return npyload!(dim)(path);
// }

/++
Get Dimension
+/
uint getDim(string path)
{
    uint dim = 0;
    auto r = regex(r"_[0-9]D_");
    foreach (d; matchAll(path, r))
    {
        dim = (d.hit[1]).to!uint - 48;
    }
    return dim;
}

/++
Implementation of an npy loader 1D
+/
auto npyload(T, uint Dim)(string path)
{
    return loadNpy!(T, 2)(path);
}
/++
Implementation of an npy loader 2D
+/
auto npyload(T, uint Dim:2)(string path)
{
    return loadNpy!(T, 3)(path);
}
/++
Implementation of an npy loader 3D
+/
auto npyload(T, uint Dim:3)(string path)
{
    return loadNpy!(T, 4)(path);
}