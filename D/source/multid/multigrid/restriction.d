module multid.multigrid.restriction;

import mir.ndslice : Slice, slice, sliced, strided, iota, as;
import std.exception : enforce;

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
    ret[0 .. end, $ - 1] = A[0 .. $, $ - 1].strided!(0)(2);
    ret[$ - 1, 0 .. end] = A[$ - 1, 0 .. $].strided!(0)(2);
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
