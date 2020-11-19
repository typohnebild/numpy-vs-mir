module multid.multigrid.restriction;

import mir.algorithm.iteration: each;
import mir.exception : enforce;
import mir.math: fastmath;
import mir.ndslice;
import numir : approxEqual;
import std.traits: isNumeric;

/++
This is the implementation of a restriction for 1D, 2D, 3D
+/
@fastmath

// TODO: move to mir.algorithm.iteration

template restriction(alias fun = "a = b")
{
    import mir.functional: naryFun;

    static if (__traits(isSame, naryFun!fun, fun))
    Slice!(T*, Dim) restriction(T, size_t Dim)(Slice!(const(T)*, Dim) A)
    {
        size_t[Dim] shape = A.length / 2 + 1;
        auto ret = shape.slice!T;
        .restriction!fun(ret, A);
        return ret;
    }

    static if (__traits(isSame, naryFun!fun, fun))
    @nogc @fastmath
    void restriction(IteratorA, IteratorB, size_t N, SliceKind aKind, SliceKind bKind)(Slice!(IteratorA, N, aKind) a, Slice!(IteratorB, N, bKind) b)
    {
        import std.traits: Select;
        alias f = Select!(N == 1, fun, .restriction!fun);
        each!f(a[0 .. $ - 1].byDim!0, b[0 .. $ - 1].byDim!0.strided(2));
        f(a.back, b.back);
    }
    else
        alias restriction = .restriction!(naryFun!fun);
}



/++
This is the implementation of a restriction for 1D, 2D, 3D
+/
@fastmath
Slice!(T*, Dim) weighted_restriction(T, size_t Dim)(Slice!(const(T)*, Dim) A)
{
    size_t[Dim] shape = A.length / 2 + 1;
    auto ret = shape.slice!T;
    weighted_restriction(ret, A);
    return ret;
}

@nogc @fastmath
void weighted_restriction(T, size_t N)(Slice!(T*, N) r, Slice!(const(T)*, N) A)
{
    weighted_restriction_borders(r, A);
    auto rc = r.canonical;
    static foreach (d; 0 .. N)
    {
        rc.popFront!d;
        rc.popBack!d;
    }
    rc.assignImpl!N = A.weights3;
}

@nogc @fastmath
void weighted_restriction_borders(T, size_t N)(Slice!(T*, N) r, Slice!(const(T)*, N) A)
{
    import std.traits: Select;
    static if (N == 1)
    {
        r.front = A.front;
        r.back = A.back;
    }
    else
    {
        restriction(r.front, A.front);
        each!weighted_restriction_borders(r[1 .. $ - 1].byDim!0, A[2 .. $ - 1].byDim!0.strided(2));
        restriction(r.back, A.back);
    }
}


private template assignImpl(size_t N)
{
    @nogc @fastmath
    void assignImpl(T)(ref T a, const T b) @fastmath
        if (isNumeric!T)
    {
        enum factor = 2 ^^ (N * 2);
        static if (__traits(isIntegral, T))
            a = b / factor;
        else
            a = b * (T(1) / (factor));
    }

    @nogc @fastmath
    void assignImpl(Slice1, Slice2)(Slice1 a, Slice2 b) @fastmath @property
        if (isSlice!Slice1)
    {
        each!assignImpl(a.byDim!0, b[1 .. $].stride(2));
    }
}

private @nogc @fastmath
auto sum3(T)(const T a, const T b, const T c) @fastmath
    if (isNumeric!T)
{
    return a + b * 2 + c;
}

private @nogc @fastmath
auto sum3(It1, It2, It3)(Slice!It1 a, Slice!It2 b, Slice!It3 c) @fastmath
{
    return zip!true(a, b, c).map!(.sum3);
}

// constructs lazy view of sums
private @nogc @fastmath
auto weights3(T, size_t N, SliceKind kind)(Slice!(const(T)*, N, kind) a) @fastmath
{
    static if (N == 1)
        return a.slide!(3, sum3);
    else
        return a.byDim!0.map!weights3.slide!(3, sum3);
}

// Test restriction 1D
unittest
{
    auto arr = iota(10).slice;
    auto ret = restriction!"a = b"(arr);
    auto correct = [0, 2, 4, 6, 8, 9];
    assert(ret == correct);

    arr = iota(11).slice;
    ret = restriction(arr);
    correct = [0, 2, 4, 6, 8, 10];
    assert(ret == correct);
}

// Test restriction 2D
unittest
{
    auto arr = [5, 5].iota.slice;
    auto ret = restriction(arr);
    auto correct = [[0, 2, 4], [10, 12, 14], [20, 22, 24]];
    assert(ret == correct);

    arr = [6, 6].iota.slice;
    ret = restriction(arr);
    correct = [[0, 2, 4, 5], [12, 14, 16, 17], [24, 26, 28, 29], [30, 32, 34, 35]];
    assert(ret == correct);
}

// Test restrtiction 3D
unittest
{
    auto arr = [5, 5, 5].iota.slice;
    auto ret = restriction(arr);
    auto correct = [[[0., 2., 4.],
            [10., 12., 14.],
            [20., 22., 24.]],

        [[50., 52., 54.],
            [60., 62., 64.],
            [70., 72., 74.]],

        [[100., 102., 104.],
            [110., 112., 114.],
            [120., 122., 124.]]];
    assert(ret == correct);

    arr = [6, 6, 6].iota.slice;
    ret = restriction(arr);
    correct = [[[0., 2., 4., 5.],
            [12., 14., 16., 17.],
            [24., 26., 28., 29.],
            [30., 32., 34., 35.]],

        [[72., 74., 76., 77.],
            [84., 86., 88., 89.],
            [96., 98., 100., 101.],
            [102., 104., 106., 107.]],

        [[144., 146., 148., 149.],
            [156., 158., 160., 161.],
            [168., 170., 172., 173.],
            [174., 176., 178., 179.]],

        [[180., 182., 184., 185.],
            [192., 194., 196., 197.],
            [204., 206., 208., 209.],
            [210., 212., 214., 215.]]];
    assert(ret == correct);
}

// Test weighted_restriction 1D long
unittest
{
    auto arr = iota(10).slice;
    auto ret = weighted_restriction(arr);
    auto correct = [0, 2, 4, 6, 8, 9];
    assert(ret == correct);

    arr = iota(11).slice;
    ret = weighted_restriction(arr);
    correct = [0, 2, 4, 6, 8, 10];
    assert(ret == correct);
}
// Test weighted_restriction 1D double
unittest
{
    auto arr2 = [1.0, 2.0, 3.0, 2.0, 1.0].sliced;
    auto ret2 = weighted_restriction(arr2);
    auto correct2 = [1.0, 2.5, 1.0];
    assert(ret2 == correct2);

    arr2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0].sliced;
    ret2 = weighted_restriction(arr2);
    correct2 = [1.0, 2.75, 2.0, 1.0];
    assert(ret2 == correct2);
}


// Test weighted_restriction 2D long
unittest
{
    auto arr = [5, 5].iota.slice;
    auto ret = weighted_restriction(arr);
    auto correct = [[0, 2, 4], [10, 12, 14], [20, 22, 24]];
    assert(ret == correct);

    arr = [6, 6].iota.slice;
    ret = restriction(arr);
    correct = [[0, 2, 4, 5], [12, 14, 16, 17], [24, 26, 28, 29], [30, 32, 34, 35]];
    assert(ret == correct);
}
// Test weighted_restriction 2D double
unittest
{
    auto arr2 = [1.0, 2.0, 3.0, 2.0, 1.0,
        2.0, 3.0, 4.0, 3.0, 2.0,
        3.0, 4.0, 5.0, 4.0, 3.0,
        4.0, 5.0, 6.0, 5.0, 4.0,
        5.0, 6.0, 7.0, 6.0, 5.0].sliced(5, 5);
    auto ret2 = weighted_restriction(arr2);
    auto correct2 = [[1.0, 3.0, 1.0],
        [3.0, 4.5, 3.0],
        [5.0, 7.0, 5.0]];

    assert(ret2 == correct2);

    arr2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0,
        2.0, 3.0, 4.0, 4.0, 3.0, 2.0,
        3.0, 4.0, 5.0, 5.0, 4.0, 3.0,
        4.0, 5.0, 6.0, 6.0, 5.0, 4.0,
        5.0, 6.0, 7.0, 7.0, 6.0, 5.0,
        6.0, 7.0, 8.0, 8.0, 7.0, 6.0].sliced(6, 6);
    ret2 = weighted_restriction(arr2);
    correct2 = [[1.0, 3.0, 2.0, 1.0],
        [3.0, 4.75, 4.0, 3.0],
        [5.0, 6.75, 6.0, 5.0],
        [6.0, 8.0, 7.0, 6.0]];
    assert(ret2 == correct2);
}

// Test weighted_restriction 3D double
unittest
{
    auto arr = [5, 5, 5].iota.as!double.slice;
    auto ret = weighted_restriction(arr);
    auto correct = [[[0., 2., 4.],
            [10., 12., 14.],
            [20., 22., 24.]],

        [[50., 52., 54.],
            [60., 62., 64.],
            [70., 72., 74.]],

        [[100., 102., 104.],
            [110., 112., 114.],
            [120., 122., 124.]]];
    assert(ret == correct);

    arr = [6, 6, 6].iota.as!double.slice;
    ret = restriction(arr);
    correct = [[[0., 2., 4., 5.],
            [12., 14., 16., 17.],
            [24., 26., 28., 29.],
            [30., 32., 34., 35.]],

        [[72., 74., 76., 77.],
            [84., 86., 88., 89.],
            [96., 98., 100., 101.],
            [102., 104., 106., 107.]],

        [[144., 146., 148., 149.],
            [156., 158., 160., 161.],
            [168., 170., 172., 173.],
            [174., 176., 178., 179.]],

        [[180., 182., 184., 185.],
            [192., 194., 196., 197.],
            [204., 206., 208., 209.],
            [210., 212., 214., 215.]]];
    assert(ret == correct);
}

unittest
{
    auto arr = [0.40088473, 0.89582552, 0.16398608, 0.45921818, 0.50720246,
        0.05841615, 0.71127485,
        0.29420544, 0.55262016, 0.45112843, 0.63388048, 0.27870701,
        0.43475406, 0.66402547,
        0.9038084, 0.16260612, 0.61827658, 0.17583573, 0.26752605,
        0.54132342, 0.95954425,
        0.21255126, 0.63423338, 0.48119557, 0.42348304, 0.66583851,
        0.80677271, 0.76529026,
        0.30096364, 0.36264674, 0.23783031, 0.21284939, 0.12336692,
        0.74549574, 0.47731472,
        0.47082754, 0.36148885, 0.24760767, 0.968772, 0.41319792,
        0.44027865, 0.65545125,
        0.98000574, 0.78919914, 0.49313159, 0.85537712, 0.44181032,
        0.4994793, 0.97419118].sliced(7, 7);
    auto correct = [0.40088473, 0.16398608, 0.50720246, 0.71127485,
        0.9038084, 0.45367844, 0.41827524, 0.95954425,
        0.30096364, 0.37174358, 0.45047108, 0.47731472,
        0.98000574, 0.49313159, 0.44181032, 0.97419118].sliced(4, 4);
    auto ret = weighted_restriction(arr);

    assert(approxEqual(ret, correct, 1e-2, 1e-8));
}

unittest
{
    auto arr = [[0.81221201, 0.78276113, 0.48331298, 0.37342158, 0.69540543,
            0.76324145, 0.82182523, 0.72875685],
        [0.57634476, 0.31967787, 0.82186108, 0.52491243, 0.15475378,
            0.13005756, 0.54944053, 0.2843028],
        [0.60829286, 0.66684961, 0.03881298, 0.36623578, 0.43896866,
            0.09926548, 0.21621183, 0.14579873],
        [0.53163999, 0.30784403, 0.79728148, 0.5986419, 0.45822312,
            0.61653698, 0.12602686, 0.84576779],
        [0.99019009, 0.15173809, 0.97024363, 0.21683838, 0.32338431,
            0.92911924, 0.76354069, 0.14346233],
        [0.90767941, 0.2732503, 0.23990377, 0.82870636, 0.66895977,
            0.55954603, 0.91480887, 0.56811022],
        [0.24181791, 0.11676617, 0.65585234, 0.74380539, 0.16570513,
            0.31648328, 0.26040337, 0.51607495],
        [0.68527047, 0.42570696, 0.13484427, 0.48563044, 0.23751941,
            0.60301295, 0.20323849, 0.59139025]];
    auto correct = [[0.81221201, 0.48331298, 0.69540543, 0.82182523, 0.72875685],
        [0.60829286, 0.450674, 0.36143624, 0.28641098, 0.14579873],
        [0.99019009, 0.54380879, 0.5277031, 0.6169349, 0.14346233],
        [0.24181791, 0.44420891, 0.44207825, 0.45405526, 0.51607495],
        [0.68527047, 0.13484427, 0.23751941, 0.20323849, 0.59139025]];
    auto ret = weighted_restriction(arr.fuse);

    assert(approxEqual(ret, correct.fuse, 1e-8, 1e-8));
}
