module multid.multigrid.restriction;

import mir.ndslice : Slice, slice, sliced, strided, iota, as, fuse;
import std.exception : enforce;
import numir: approxEqual;

/++
This is the implementation of a restriction for 1D
+/
Slice!(T*, Dim) restriction(T, size_t Dim : 1)(Slice!(T*, Dim) A)
{
    auto N = A.shape[0] / 2 + 1;
    auto ret = slice!T([N], 0);
    const auto end = N - (A.shape[0] + 1) % 2;
    auto AF = A.field;

    foreach (i; 0 .. end)
    {
        // get every second element in AF
        ret.field[i] = AF[i * 2];
    }
    // special case: outer corner
    ret.field[$ - 1] = AF[$ - 1];

    return ret;
}

/++
This is the implementation of a restriction for 2D
works only for square grids
+/
Slice!(T*, Dim) restriction(T, size_t Dim : 2)(Slice!(T*, Dim) A)
{
    enforce(A.shape[0] == A.shape[1], "not all dimensions have the same length");
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = slice!T([N, N], 0);
    const auto end = N - (A.shape[0] + 1) % 2;
    auto AF = A.field;

    foreach (i; 0 .. end)
    {
        foreach (j; 0 .. end)
        {
            // get every second element in AF
            auto flattindexret = i * N + j;
            auto flattindexA = i * M * 2 + j * 2;
            ret.field[flattindexret] = AF[flattindexA];
        }
    }
    // special case: borders
    // ret[0 .. end, $ - 1] = A[0 .. $, $ - 1].strided!(0)(2);
    // ret[$ - 1, 0 .. end] = A[$ - 1, 0 .. $].strided!(0)(2);
    foreach (i; 0..end)
    {
        auto indexrowR = (N-1)*N + i;
        auto indexrowA = (M-1)*M + 2 * i;
        auto indexcolR = N * i + N - 1;
        auto indexcolA = M * 2 * i + M - 1;
        ret.field[indexrowR] = AF[indexrowA];
        ret.field[indexcolR] = AF[indexcolA];
    }
    // special case: outer corner
    ret.field[$ - 1] = AF[$ - 1];

    return ret;
}

/++
This is the implementation of a restriction for 3D
works only if all dimensions have the same length
+/
Slice!(T*, Dim) restriction(T, size_t Dim : 3)(Slice!(T*, Dim) A)
{
    enforce(A.shape[0] == A.shape[1] && A.shape[1] == A.shape[2], "not all dimensions have the same length");
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = slice!T([N, N, N], 0);
    const auto end = N - (A.shape[0] + 1) % 2;
    auto AF = A.field;

    foreach (k; 0 .. end)
    {
        foreach (i; 0 .. end)
        {
            foreach (j; 0 .. end)
            {
                // get every second element in AF
                auto flattindexret = k * (N * N) + i * N + j;
                auto flattindexA = k * (M * M) * 2 + i * M * 2 + j * 2;
                ret.field[flattindexret] = AF[flattindexA];
            }
        }
    }
    // special case: inner borders
    ret[0 .. end, 0 .. end, $ - 1] = A[0 .. $, 0 .. $, $ - 1].strided!(0, 1)(2, 2);
    ret[$ - 1, 0 .. end, 0 .. end] = A[$ - 1, 0 .. $, 0 .. $].strided!(0, 1)(2, 2);
    ret[0 .. end, $ - 1, 0 .. end] = A[0 .. $, $ - 1, 0 .. $].strided!(0, 1)(2, 2);
    // special case: outer borders
    ret[0 .. end, $ - 1, $ - 1] = A[0 .. $, $ - 1, $ - 1].strided!(0)(2);
    ret[$ - 1, 0 .. end, $ - 1] = A[$ - 1, 0 .. $, $ - 1].strided!(0)(2);
    ret[$ - 1, $ - 1, 0 .. end] = A[$ - 1, $ - 1, 0 .. $].strided!(0)(2);
    // special case: outer corner
    ret.field[$ - 1] = AF[$ - 1];

    return ret;
}

/++
This is the implementation of a weighted_restriction 1D
+/
Slice!(T*, Dim) weighted_restriction(T, size_t Dim : 1)(Slice!(T*, Dim) A)
{
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = restriction!(T, Dim)(A);
    auto AF = A.field;
    foreach (i; 1u .. N - 1u)
    {
        ret.field[i] = ret.field[i] / 2 + (AF[i * 2 - 1u] + AF[i * 2 + 1u]) / cast(T)(4);
    }
    return ret;
}

/++
This is the implementation of a weighted_restriction 2D
+/
Slice!(T*, Dim) weighted_restriction(T, size_t Dim : 2)(Slice!(T*, Dim) A)
{
    enforce(A.shape[0] == A.shape[1], "not all dimensions have the same length");
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = restriction!(T, Dim)(A);
    auto AF = A.field;

    foreach (i; 1u .. N - 1u)
    {
        foreach (j; 1u .. N - 1u)
        {
            auto indexR = i * N + j;
            auto indexA = i * M * 2 + j * 2;
            ret.field[indexR] = ret.field[indexR] / cast(T)(4) +
                (
                        AF[indexA - 1] + AF[indexA + 1] +
                        AF[indexA - M] + AF[indexA + M]) / cast(T)(8) +
                (
                        AF[indexA - 1 - M] + AF[indexA - 1 + M] +
                        AF[indexA + 1 - M] + AF[indexA + 1 + M]) / cast(T)(16);
        }
    }
    return ret;
}

/++
This is the implementation of a weighted_restriction 3D
+/
Slice!(T*, Dim) weighted_restriction(T, size_t Dim : 3)(Slice!(T*, Dim) A)
{
    enforce(A.shape[0] == A.shape[1] && A.shape[1] == A.shape[2], "not all dimensions have the same length");
    const auto M = A.shape[0];
    const auto N = M / 2 + 1;
    auto ret = restriction!(T, Dim)(A);
    auto AF = A.field;
    foreach (k; 1u .. N - 1u)
    {
        foreach (i; 1u .. N - 1u)
        {
            foreach (j; 1u .. N - 1u)
            {
                auto indexR = k * (N * N) + i * N + j;
                auto indexA = k * (M * M) * 2 + i * M * 2 + j * 2;
                ret.field[indexR] = (
                        ret.field[indexR] * cast(T)(8) +
                        (
                            AF[indexA - 1] + AF[indexA + 1] +
                            AF[indexA - M] + AF[indexA + M] +
                            AF[indexA - M * M] + AF[indexA + M * M]) * cast(T)(4) +
                        (
                            AF[indexA - 1 - M] + AF[indexA + 1 + M] +
                            AF[indexA - 1 + M] + AF[indexA + 1 - M] +
                            AF[indexA - 1 - M * M] + AF[indexA + 1 + M * M] +
                            AF[indexA - 1 + M * M] + AF[indexA + 1 - M * M] +
                            AF[indexA - M - M * M] + AF[indexA + M + M * M] +
                            AF[indexA - M + M * M] + AF[indexA + M - M * M]) * cast(
                            T)(2) +
                        (AF[indexA - 1 - M - M * M] + AF[indexA + 1 - M - M * M] +
                            AF[indexA - 1 + M - M * M] + AF[indexA + 1 + M - M * M] +
                            AF[indexA - 1 - M + M * M] + AF[indexA + 1 - M + M * M] +
                            AF[indexA - 1 + M + M * M] + AF[indexA + 1 + M + M * M])) /
                    cast(T)(64);
            }
        }
    }
    return ret;
}

// Test restriction 1D
unittest
{
    auto arr = iota(10).slice;
    auto ret = restriction!(long, 1)(arr);
    auto correct = [0, 2, 4, 6, 8, 9];
    assert(ret == correct);

    arr = iota(11).slice;
    ret = restriction!(long, 1)(arr);
    correct = [0, 2, 4, 6, 8, 10];
    assert(ret == correct);
}

// Test restriction 2D
unittest
{
    auto arr = iota([5, 5]).slice;
    auto ret = restriction!(long, 2)(arr);
    auto correct = [[0, 2, 4], [10, 12, 14], [20, 22, 24]];
    assert(ret == correct);

    arr = iota([6, 6]).slice;
    ret = restriction!(long, 2)(arr);
    correct = [[0, 2, 4, 5], [12, 14, 16, 17], [24, 26, 28, 29], [30, 32, 34, 35]];
    assert(ret == correct);
}

// Test restrtiction 3D
unittest
{
    auto arr = iota([5, 5, 5]).slice;
    auto ret = restriction!(long, 3)(arr);
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

    arr = iota([6, 6, 6]).slice;
    ret = restriction!(long, 3)(arr);
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
    auto ret = weighted_restriction!(long, 1)(arr);
    auto correct = [0, 2, 4, 6, 8, 9];
    assert(ret == correct);

    arr = iota(11).slice;
    ret = weighted_restriction!(long, 1)(arr);
    correct = [0, 2, 4, 6, 8, 10];
    assert(ret == correct);
}
// Test weighted_restriction 1D double
unittest
{
    auto arr2 = [1.0, 2.0, 3.0, 2.0, 1.0].sliced;
    auto ret2 = weighted_restriction!(double, 1)(arr2);
    auto correct2 = [1.0, 2.5, 1.0];
    assert(ret2 == correct2);

    arr2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0].sliced;
    ret2 = weighted_restriction!(double, 1)(arr2);
    correct2 = [1.0, 2.75, 2.0, 1.0];
    assert(ret2 == correct2);
}

// Test weighted_restriction 2D long
unittest
{
    auto arr = iota([5, 5]).slice;
    auto ret = weighted_restriction!(long, 2)(arr);
    auto correct = [[0, 2, 4], [10, 12, 14], [20, 22, 24]];
    assert(ret == correct);

    arr = iota([6, 6]).slice;
    ret = restriction!(long, 2)(arr);
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
    auto ret2 = weighted_restriction!(double, 2)(arr2);
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
    ret2 = weighted_restriction!(double, 2)(arr2);
    correct2 = [[1.0, 3.0, 2.0, 1.0],
        [3.0, 4.75, 4.0, 3.0],
        [5.0, 6.75, 6.0, 5.0],
        [6.0, 8.0, 7.0, 6.0]];
    assert(ret2 == correct2);
}

// Test weighted_restriction 3D double
unittest
{
    auto arr = iota([5, 5, 5]).as!double.slice;
    auto ret = weighted_restriction!(double, 3)(arr);
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

    arr = iota([6, 6, 6]).as!double.slice;
    ret = restriction!(double, 3)(arr);
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
       0.9038084 , 0.16260612, 0.61827658, 0.17583573, 0.26752605,
        0.54132342, 0.95954425,
       0.21255126, 0.63423338, 0.48119557, 0.42348304, 0.66583851,
        0.80677271, 0.76529026,
       0.30096364, 0.36264674, 0.23783031, 0.21284939, 0.12336692,
        0.74549574, 0.47731472,
       0.47082754, 0.36148885, 0.24760767, 0.968772  , 0.41319792,
        0.44027865, 0.65545125,
       0.98000574, 0.78919914, 0.49313159, 0.85537712, 0.44181032,
        0.4994793 , 0.97419118].sliced(7, 7);
    auto correct = [0.40088473, 0.16398608, 0.50720246, 0.71127485,
       0.9038084 , 0.45367844, 0.41827524, 0.95954425,
       0.30096364, 0.37174358, 0.45047108, 0.47731472,
       0.98000574, 0.49313159, 0.44181032, 0.97419118].sliced(4,4);
    auto ret = weighted_restriction!(double, 2)(arr);

    assert (approxEqual(ret, correct, 1e-2, 1e-8));
}

unittest
{
    auto arr = [[0.81221201, 0.78276113, 0.48331298, 0.37342158, 0.69540543,
        0.76324145, 0.82182523, 0.72875685],
       [0.57634476, 0.31967787, 0.82186108, 0.52491243, 0.15475378,
        0.13005756, 0.54944053, 0.2843028 ],
       [0.60829286, 0.66684961, 0.03881298, 0.36623578, 0.43896866,
        0.09926548, 0.21621183, 0.14579873],
       [0.53163999, 0.30784403, 0.79728148, 0.5986419 , 0.45822312,
        0.61653698, 0.12602686, 0.84576779],
       [0.99019009, 0.15173809, 0.97024363, 0.21683838, 0.32338431,
        0.92911924, 0.76354069, 0.14346233],
       [0.90767941, 0.2732503 , 0.23990377, 0.82870636, 0.66895977,
        0.55954603, 0.91480887, 0.56811022],
       [0.24181791, 0.11676617, 0.65585234, 0.74380539, 0.16570513,
        0.31648328, 0.26040337, 0.51607495],
       [0.68527047, 0.42570696, 0.13484427, 0.48563044, 0.23751941,
        0.60301295, 0.20323849, 0.59139025]];
    auto  correct = [[0.81221201, 0.48331298, 0.69540543, 0.82182523, 0.72875685],
       [0.60829286, 0.450674  , 0.36143624, 0.28641098, 0.14579873],
       [0.99019009, 0.54380879, 0.5277031 , 0.6169349 , 0.14346233],
       [0.24181791, 0.44420891, 0.44207825, 0.45405526, 0.51607495],
       [0.68527047, 0.13484427, 0.23751941, 0.20323849, 0.59139025]];
    auto ret = weighted_restriction!(double, 2)(arr.fuse);

    assert (approxEqual(ret, correct.fuse, 1e-8, 1e-8));
}