import std.stdio;
import std.range: generate;
import std.random: uniform;
import std.array;
import std.algorithm;
import mir.ndslice;

import multid.gaussseidel.redblack;


void main()
{
    auto F = slice!double(5, 2);
    auto fun = generate!(() => uniform(0.0, 1.0));
    F.field.fill(fun);
    auto U = slice!double([5,2], 0.0);
    double h = 1;
    // auto F = generate!(() => uniform(0, 0.99)).take(10).array.sliced;
    // F = F.reshape([5,2]);
    writeln(GS_RB!(double, 2, 666)(F, U, h));
}
