module multid.multigrid.restriction;

import std.stdio;
import mir.ndslice;

/++
This is the implementation of a restriction for 1D
+/
Slice!(T*, Dim) restriction(T, size_t Dim:1)(Slice!(T*, Dim) A)
{
    auto N = A.shape[0] / 2 + 1;
    auto ret = slice!T([N], 0);
    const auto end = N - (A.shape[0] + 1) % 2;
    auto AF = A.field;

    foreach (i; 0 .. end)
    {
        // get every second element in AF
        ret.field[i] = AF[i*2];
    }
    // special case: outer corner
    ret.field[$-1] = AF[$-1];

    return ret;
}

/++
This is the implementation of a restriction for 2D
+/
Slice!(T*, Dim) restriction(T, size_t Dim:2)(Slice!(T*, Dim) A)
{
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = slice!T([N, N], 0);
    const auto end = N - (A.shape[0] + 1) % 2;
    auto AF = A.field;

    foreach (i; 0 .. end)
    {
        foreach (j; 0..end)
        {
            // get every second element in AF
            auto flattindexret = i * N + j;
            auto flattindexA = i * M * 2 + j * 2;
            ret.field[flattindexret] = AF[flattindexA];
        }
    }
    // special case: borders
    ret[0..end, $-1] = A[0 .. $, $-1].strided!(0)(2);
    ret[$-1, 0..end] = A[$-1, 0..$].strided!(0)(2);
    // special case: outer corner
    ret.field[$-1] = AF[$-1];

    return ret;
}

/++
This is the implementation of a restriction for 3D
+/
Slice!(T*, Dim) restriction(T, size_t Dim:3)(Slice!(T*, Dim) A)
{
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = slice!T([N, N, N], 0);
    const auto end = N - (A.shape[0] + 1) % 2;
    auto AF = A.field;

    foreach (k; 0 .. end)
    {
        foreach (i; 0 .. end)
        {
            foreach (j; 0..end)
            {
                // get every second element in AF
                auto flattindexret = k*(N*N) + i * N + j;
                auto flattindexA = k*(M*M)*2 + i * M * 2 + j * 2;
                ret.field[flattindexret] = AF[flattindexA];
            }
        }
    }
    // special case: inner borders
    ret[0..end, 0..end, $-1] = A[0..$, 0..$, $-1].strided!(0,1)(2,2);
    ret[$-1, 0..end, 0..end] = A[$-1, 0..$, 0..$].strided!(0,1)(2,2);
    ret[0..end, $-1, 0..end] = A[0..$, $-1, 0..$].strided!(0,1)(2,2);
    // special case: outer borders
    ret[0..end, $-1, $-1] = A[0..$, $-1, $-1].strided!(0)(2);
    ret[$-1, 0..end, $-1] = A[$-1, 0..$, $-1].strided!(0)(2);
    ret[$-1, $-1, 0..end] = A[$-1, $-1, 0..$].strided!(0)(2);
    // special case: outer corner
    ret.field[$-1] = AF[$-1];

    return ret;
}

/++
This is the implementation of a weighted_restriction
+/
void weighted_restriction(T, size_t Dim)(Slice!(T*, Dim) A)
{
    //TODO

    static if (Dim == 1)
    {
        //TODO
    }
    else static if (Dim == 2)
    {
        //TODO
    }
    else static if (Dim == 3)
    {
        //TODO
    }
    else
    {
        static assert(false, Dim.stringof ~ " is not a supported dimension!");
    }
}





// Test restriction 1D
unittest
{
    auto arr = iota(10).slice;
    auto ret = restriction!(long, 1)(arr);
    auto correct = [0,2,4,6,8,9];
    assert (ret == correct);

    arr = iota(11).slice;
    ret = restriction!(long, 1)(arr);
    correct = [0,2,4,6,8,10];
    assert (ret == correct);
}

// Test restriction 2D
unittest
{
    auto arr = iota([5,5]).slice;
    auto ret = restriction!(long, 2)(arr);
    auto correct = [[0,2,4],[10,12,14],[20,22,24]];
    assert(ret == correct);

    arr = iota([6,6]).slice;
    ret = restriction!(long,2)(arr);
    correct = [[0,2,4,5], [12,14,16,17], [24,26,28,29], [30,32,34,35]];
    assert(ret == correct);
}

// Test restrtiction 3D
unittest
{
    auto arr = iota([5, 5, 5]).slice;
    auto ret = restriction!(long, 3)(arr);
    auto correct = [[[  0.,   2.,   4.],
        [ 10.,  12.,  14.],
        [ 20.,  22.,  24.]],

       [[ 50.,  52.,  54.],
        [ 60.,  62.,  64.],
        [ 70.,  72.,  74.]],

       [[100., 102., 104.],
        [110., 112., 114.],
        [120., 122., 124.]]];
    assert(ret == correct);

    arr = iota([6, 6, 6]).slice;
    ret = restriction!(long, 3)(arr);
    correct = [[[  0.,   2.,   4.,   5.],
        [ 12.,  14.,  16.,  17.],
        [ 24.,  26.,  28.,  29.],
        [ 30.,  32.,  34.,  35.]],

       [[ 72.,  74.,  76.,  77.],
        [ 84.,  86.,  88.,  89.],
        [ 96.,  98., 100., 101.],
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