module multid.multigrid.prolongation;

import mir.ndslice : slice, Slice;
import numir : approxEqual;

/++
This is the implementation of a prolongation
    Params:
        e = the grid that needs to be prolongated
        fine_shape = the shape of the returned grid
    Returns: the finer grid with interpolated values in between
+/
Slice!(T*, Dim) prolongation(T, size_t Dim)(in Slice!(T*, Dim) e, in size_t[Dim] fine_shape)
{
    auto w = slice!T(fine_shape);
    auto end = e.shape[0] - (fine_shape[0] + 1) % 2;
    const auto wend = w.shape[0] - (fine_shape[0] + 1) % 2;
    auto WF = w.field;
    auto EF = e.field;

    static if (Dim == 1)
    {
        for (size_t i = 1; i < end; i++)
        {
            w.field[2 * i - 1] = (e.field[i - 1] + e.field[i]) / 2;
            w.field[2 * i] = e.field[i];
        }
        w.field[$ - 1] = e.field[$ - 1];
        w.field[0] = e.field[0];
    }
    else static if (Dim == 2)
    {
        foreach (i; 0 .. end - 1)
        {
            auto flatindexw = 2 * i * w.shape[0];
            auto flatindexw2 = (2 * i + 1) * w.shape[0];
            auto flatindexe = i * e.shape[0];
            auto flatindexe2 = (i + 1) * e.shape[0];
            foreach (j; 0 .. end - 1)
            {
                // the value that is copied
                WF[flatindexw + 2 * j] = EF[flatindexe + j];
                // the value next a copied one
                WF[flatindexw + 2 * (j + 1) - 1] = (EF[flatindexe + j] + EF[flatindexe + j + 1]) / 2;
                // the value below a copied one
                WF[flatindexw2 + 2 * j] = (EF[flatindexe2 + j] + EF[flatindexe + j]) / 2;
            }
            WF[flatindexw + 2 * (end - 1)] = EF[flatindexe + end - 1];
            WF[flatindexw2 + 2 * (end - 1)] = (EF[flatindexe2 + end - 1] +
                    EF[flatindexe + end - 1]) / 2;
        }
        // this is for the last row and the last colomn
        auto flatindexw = 2 * (end - 1) * w.shape[0];
        auto flatindexe = (end - 1) * e.shape[0];
        foreach (j; 0 .. end - 1)
        {
            WF[flatindexw + 2 * j] = EF[flatindexe + j];
            WF[flatindexw + 2 * (j + 1) - 1] = (EF[flatindexe + j] + EF[flatindexe + j + 1]) / 2;
        }
        WF[$ - 1] = EF[$ - 1];
        for (size_t i = 1; i < wend; i += 2)
        {
            for (size_t j = 1; j < wend; j += 2)
            {
                auto flatindex = i * w.shape[0] + j;
                WF[flatindex] = (
                        WF[flatindex + w.shape[0]] + WF[flatindex -
                        w.shape[0]] + WF[flatindex - 1] + WF[flatindex + 1]) / 4;
            }
        }
        // Since we restrict always to N//2 + 1 we need to handle the case if
        // the finer grid is even sized, because that means between the last
        // and the forelast is no new colomn that needs to be calculated
        if (fine_shape[0] % 2 == 0) // != e.shape[0] % 2)
        {
            flatindexw = (w.shape[0] - 1) * w.shape[0];
            flatindexe = (e.shape[0] - 1) * e.shape[0];
            foreach (j; 0 .. end - 1)
            {
                WF[flatindexw + 2 * j] = EF[flatindexe + j];
                WF[flatindexw + 2 * (j + 1) - 1] = (EF[flatindexe + j] + EF[flatindexe + j + 1]) / 2;

                WF[(w.shape[0]) * 2 * j + w.shape[0] - 1] = EF[(
                            e.shape[0]) * j + e.shape[0] - 1];

                WF[w.shape[0] * (2 * j + 1) + w.shape[0] - 1] = (
                        EF[e.shape[0] * j + e.shape[0] - 1] +
                        EF[e.shape[0] * (j + 1) + e.shape[0] - 1]) / 2;
            }
            w[$ - 2 .. $, $ - 2 .. $] = e[$ - 2 .. $, $ - 2 .. $];

        }

    }
    else static if (Dim == 3)
    {
        //TODO
        immutable MW = fine_shape[0];
        immutable NW = fine_shape[1];
        immutable OW = fine_shape[2];

        immutable ME = e.shape[0];
        immutable NE = e.shape[1];
        immutable OE = e.shape[2];

        auto idxe = (size_t i, size_t j, size_t k) => i * (ME * NE) + j * NE + k;
        auto idxw = (size_t i, size_t j, size_t k) => i * (MW * NW) + j * NW + k;

        WF[$ - 1] = EF[$ - 1];

        foreach (i; 0 .. end - 1)
        {
            foreach (j; 0 .. end - 1)
            {
                foreach (k; 0 .. end - 1)
                {
                    WF[idxw(2 * i, 2 * j, 2 * k)] = EF[idxe(i, j, k)];

                    WF[idxw(2 * i, 2 * j, 2 * k + 1)] = (EF[idxe(i, j, k)] + EF[idxe(i, j, k + 1)]) / 2;
                    WF[idxw(2 * i, 2 * j + 1, 2 * k)] = (EF[idxe(i, j, k)] + EF[idxe(i, j + 1, k)]) / 2;

                    WF[idxw(2 * i + 1, 2 * j, 2 * k)] = (EF[idxe(i, j, k)] + EF[idxe(i + 1, j, k)]) / 2;

                    WF[idxw(2 * i, 2 * j + 1, 2 * k + 1)] = (
                            EF[idxe(i, j, k)] +
                            EF[idxe(i, j + 1, k)] +
                            EF[idxe(i, j, k + 1)] +
                            EF[idxe(i, j + 1, k + 1)]) / 4;

                    WF[idxw(2 * i + 1, 2 * j + 1, 2 * k)] = (
                            EF[idxe(i, j, k)] +
                            EF[idxe(i, j + 1, k)] +
                            EF[idxe(i + 1, j, k)] +
                            EF[idxe(i + 1, j + 1, k)]) / 4;

                    WF[idxw(2 * i + 1, 2 * j, 2 * k + 1)] = (
                            EF[idxe(i, j, k)] +
                            EF[idxe(i, j, k + 1)] +
                            EF[idxe(i + 1, j, k)] +
                            EF[idxe(i + 1, j, k + 1)]) / 4;

                    WF[idxw(2 * i + 1, 2 * j + 1, 2 * k + 1)] = (
                            EF[idxe(i, j, k)] +
                            EF[idxe(i, j + 1, k)] +
                            EF[idxe(i, j, k + 1)] +
                            EF[idxe(i, j + 1, k + 1)] +
                            EF[idxe(i + 1, j, k)] +
                            EF[idxe(i + 1, j, k + 1)] +
                            EF[idxe(i + 1, j + 1, k)] +
                            EF[idxe(i + 1, j + 1, k + 1)]) / 8;

                }

                WF[idxw(2 * i, 2 * j, 2 * (end - 1))] = EF[idxe(i, j, end - 1)];

                WF[idxw(2 * i, 2 * j + 1, 2 * (end - 1))] = (
                        EF[idxe(i, j, (end - 1))] + EF[idxe(i, j + 1, (end - 1))]) / 2;

                WF[idxw(2 * i, 2 * (end - 1), 2 * j + 1)] = (
                        EF[idxe(i, (end - 1), j)] + EF[idxe(i, (end - 1), j + 1)]) / 2;

                WF[idxw(2 * (end - 1), 2 * i, 2 * j)] = EF[idxe(end - 1, i, j)];

                WF[idxw(2 * (end - 1), 2 * i, 2 * j + 1)] = (
                        EF[idxe(end - 1, i, j)] + EF[idxe(end - 1, i, j + 1)]) / 2;

                WF[idxw(2 * i, 2 * (end - 1), 2 * j)] = EF[idxe(i, end - 1, j)];

                WF[idxw(2 * (end - 1), 2 * i + 1, 2 * j)] = (
                        EF[idxe(end - 1, i, j)] + EF[idxe(end - 1, i + 1, j)]) / 2;

                WF[idxw(2 * i + 1, 2 * j, 2 * (end - 1))] = (
                        EF[idxe(i, j, (end - 1))] + EF[idxe(i + 1, j, (end - 1))]) / 2;

                WF[idxw(2 * i + 1, 2 * (end - 1), 2 * j)] = (
                        EF[idxe(i, (end - 1), j)] + EF[idxe(i + 1, (end - 1), j)]) / 2;

                WF[idxw(2 * (end - 1), 2 * i + 1, 2 * j + 1)] = (
                        EF[idxe((end - 1), i, j)] +
                        EF[idxe((end - 1), i + 1, j)] +
                        EF[idxe((end - 1), i, j + 1)] +
                        EF[idxe((end - 1), i + 1, j + 1)]) / 4;

                WF[idxw(2 * i + 1, 2 * j + 1, 2 * (end - 1))] = (
                        EF[idxe(i, j, (end - 1))] +
                        EF[idxe(i, j + 1, (end - 1))] +
                        EF[idxe(i + 1, j, (end - 1))] +
                        EF[idxe(i + 1, j + 1, (end - 1))]) / 4;

                WF[idxw(2 * i + 1, 2 * (end - 1), 2 * j + 1)] = (
                        EF[idxe(i, (end - 1), j)] +
                        EF[idxe(i, (end - 1), j + 1)] +
                        EF[idxe(i + 1, (end - 1), j)] +
                        EF[idxe(i + 1, (end - 1), j + 1)]) / 4;

            }

            WF[idxw(2 * i, 2 * (end - 1), 2 * (end - 1))] = EF[idxe(i, end - 1, end - 1)];

            WF[idxw(2 * i + 1, 2 * (end - 1), 2 * (end - 1))] = (
                    EF[idxe(i, end - 1, end - 1)] + EF[idxe(i + 1, end - 1, end - 1)]) / 2;

            WF[idxw(2 * (end - 1), 2 * i, 2 * (end - 1))] = EF[idxe(end - 1, i, end - 1)];

            WF[idxw(2 * (end - 1), 2 * i + 1, 2 * (end - 1))] = (
                    EF[idxe(end - 1, i, end - 1)] + EF[idxe(end - 1, i + 1, end - 1)]) / 2;

            WF[idxw(2 * (end - 1), 2 * (end - 1), 2 * i)] = EF[idxe(end - 1, end - 1, i)];

            WF[idxw(2 * (end - 1), 2 * (end - 1), 2 * i + 1)] = (
                    EF[idxe(end - 1, end - 1, i)] + EF[idxe(end - 1, end - 1, i + 1)]) / 2;
        }
        // Since we restrict always to N//2 + 1 we need to handle the case if
        // the finer grid is even sized, because that means between the last
        // and the forelast is no new colomn that needs to be calculated
        if (fine_shape[0] % 2 == 0) // != e.shape[0] % 2)
        {

            foreach (i; 0 .. end - 1)
            {
                foreach (j; 0 .. end - 1)
                {
                    WF[idxw(2 * i, 2 * j, OW - 1)] = EF[idxe(i, j, OE - 1)];
                    WF[idxw(2 * i, MW - 1, 2 * j)] = EF[idxe(i, ME - 1, j)];
                    WF[idxw(NW - 1, 2 * i, 2 * j)] = EF[idxe(NE - 1, i, j)];

                    WF[idxw(2 * i, 2 * j + 1, OW - 1)] = (
                            EF[idxe(i, j, OE - 1)] +
                            EF[idxe(i, j + 1, OE - 1)]) / 2;

                    WF[idxw(2 * i, MW - 1, 2 * j + 1)] = (
                            EF[idxe(i, ME - 1, j)] +
                            EF[idxe(i, ME - 1, j + 1)]) / 2;

                    WF[idxw(2 * i + 1, 2 * j, OW - 1)] = (
                            EF[idxe(i, j, OE - 1)] +
                            EF[idxe(i + 1, j, OE - 1)]) / 2;

                    WF[idxw(2 * i + 1, MW - 1, 2 * j)] = (
                            EF[idxe(i, ME - 1, j)] +
                            EF[idxe(i + 1, ME - 1, j)]) / 2;

                    WF[idxw(2 * i + 1, MW - 1, 2 * j + 1)] = (
                            EF[idxe(i, ME - 1, j)] +
                            EF[idxe(i + 1, ME - 1, j)] +
                            EF[idxe(i, ME - 1, j + 1)] +
                            EF[idxe(i + 1, ME - 1, j + 1)]) / 4;

                    WF[idxw(2 * i + 1, 2 * j + 1, OW - 1)] = (
                            EF[idxe(i, j, OE - 1)] +
                            EF[idxe(i, j + 1, OE - 1)] +
                            EF[idxe(i + 1, j, OE - 1)] +
                            EF[idxe(i + 1, j + 1, OE - 1)]) / 4;

                    WF[idxw(NW - 1, 2 * i + 1, 2 * j)] = (
                            EF[idxe(NE - 1, i, j)] +
                            EF[idxe(NE - 1, i + 1, j)]) / 2;

                    WF[idxw(NW - 1, 2 * i, 2 * j + 1)] = (
                            EF[idxe(NE - 1, i, j)] +
                            EF[idxe(NE - 1, i, j + 1)]) / 2;

                    WF[idxw(NW - 1, 2 * i + 1, 2 * j + 1)] = (
                            EF[idxe(NE - 1, i, j)] +
                            EF[idxe(NE - 1, i, j + 1)] +
                            EF[idxe(NE - 1, i + 1, j)] +
                            EF[idxe(NE - 1, i + 1, j + 1)]) / 4;
                }

                WF[idxw(2 * i, 2 * (end - 1), OW - 1)] = EF[idxe(i, (end - 1), OE - 1)];
                WF[idxw(2 * i, MW - 1, 2 * (end - 1))] = EF[idxe(i, ME - 1, (end - 1))];

                WF[idxw(2 * (end - 1), MW - 1, 2 * i)] = EF[idxe((end - 1), ME - 1, i)];
                WF[idxw(NW - 1, 2 * (end - 1), 2 * i)] = EF[idxe(NE - 1, (end - 1), i)];

                WF[idxw(NW - 1, 2 * i, 2 * (end - 1))] = EF[idxe(NE - 1, i, (end - 1))];
                WF[idxw(2 * (end - 1), 2 * i, OW - 1)] = EF[idxe((end - 1), i, OE - 1)];

                WF[idxw(2 * i, MW - 1, OW - 1)] = EF[idxe(i, ME - 1, OE - 1)];
                WF[idxw(NW - 1, 2 * i, OW - 1)] = EF[idxe(NE - 1, i, OE - 1)];
                WF[idxw(NW - 1, MW - 1, 2 * i)] = EF[idxe(NE - 1, ME - 1, i)];

                WF[idxw(2 * (end - 1), MW - 1, 2 * i + 1)] = (
                        EF[idxe((end - 1), ME - 1, i)] +
                        EF[idxe((end - 1), ME - 1, i + 1)]) / 2;

                WF[idxw(2 * (end - 1), 2 * i + 1, OW - 1)] = (
                        EF[idxe((end - 1), i, OE - 1)] +
                        EF[idxe((end - 1), i + 1, OE - 1)]) / 2;

                WF[idxw((NW - 1), 2 * i + 1, OW - 1)] = (
                        EF[idxe((NE - 1), i, OE - 1)] +
                        EF[idxe((NE - 1), i + 1, OE - 1)]) / 2;

                WF[idxw(NW - 1, MW - 1, 2 * i + 1)] = (
                        EF[idxe(NE - 1, ME - 1, i)] +
                        EF[idxe(NE - 1, ME - 1, i + 1)]) / 2;

                WF[idxw(2 * i + 1, MW - 1, OW - 1)] = (
                        EF[idxe(i, ME - 1, OE - 1)] +
                        EF[idxe(i + 1, ME - 1, OE - 1)]) / 2;

                WF[idxw(2 * i + 1, 2 * (end - 1), OW - 1)] = (
                        EF[idxe(i, (end - 1), OE - 1)] +
                        EF[idxe(i + 1, (end - 1), OE - 1)]) / 2;

                WF[idxw(2 * i + 1, MW - 1, 2 * (end - 1))] = (EF[idxe(i, ME - 1, (end - 1))] +
                        EF[idxe(i + 1, ME - 1, (end - 1))]) / 2;

                WF[idxw(NW - 1, 2 * i + 1, 2 * (end - 1))] = (
                        EF[idxe(NE - 1, i, (end - 1))] +
                        EF[idxe(NE - 1, i + 1, (end - 1))]) / 2;

                WF[idxw(NW - 1, 2 * (end - 1), 2 * i + 1)] = (
                        EF[idxe(NE - 1, (end - 1), i)] +
                        EF[idxe(NE - 1, (end - 1), i + 1)]) / 2;
            }

            WF[idxw(2 * (end - 1), MW - 1, OW - 1)] = EF[idxe((end - 1), ME - 1, OE - 1)];
            WF[idxw(NW - 1, 2 * (end - 1), OW - 1)] = EF[idxe(NE - 1, (end - 1), OE - 1)];
            WF[idxw(NW - 1, MW - 1, 2 * (end - 1))] = EF[idxe(NE - 1, ME - 1, (end - 1))];

            WF[idxw(2 * (end - 1), MW - 1, 2 * (end - 1))] = EF[idxe((end - 1), ME - 1, (end - 1))];
            WF[idxw(2 * (end - 1), 2 * (end - 1), OW - 1)] = EF[idxe((end - 1), (end - 1), OE - 1)];
            WF[idxw(NW - 1, 2 * (end - 1), 2 * (end - 1))] = EF[idxe(NE - 1, end - 1, (end - 1))];
            WF[idxw(2 * (end - 1), 2 * (end - 1), 2 * (end - 1))] =
                EF[idxe((end - 1), (end - 1), (end - 1))];

        }

    }
    else
    {
        static assert(false, Dim.stringof ~ " is not a supported dimension!");
    }
    return w;
}

// Tests 1D
unittest
{
    import mir.ndslice : iota, sliced;

    auto a = [0, 2, 4, 6, 8].sliced!double;
    auto correct = 9.iota.slice;
    const auto ret = prolongation!(double, 1)(a, correct.shape);
    assert(ret == correct);

    auto a2 = [0, 2, 4, 6, 8, 9].sliced!long;
    auto correct2 = 10.iota.slice;
    const auto ret2 = prolongation!(long, 1)(a2, correct2.shape);
    assert(ret2 == correct2);

    auto a3 = [0, 2, 4, 6, 7].sliced!long;
    auto correct3 = 8.iota.slice;
    const auto ret3 = prolongation!(long, 1)(a3, correct3.shape);
    assert(ret3 == correct3);

}

// Tests 2D
unittest
{
    import mir.ndslice : iota, sliced;

    auto arr = [
        0., 2., 4., 6., 8.,
        18., 20., 22., 24., 26.,
        36., 38., 40., 42., 44.,
        54., 56., 58., 60., 62.,
        72., 74., 76., 78., 80.
    ].sliced(5, 5);

    auto correct = iota([9, 9]).slice;
    const auto ret = prolongation!(double, 2)(arr, correct.shape);
    assert(ret == correct);
    auto arr2 = [0., 2., 4., 6., 7., 16., 18., 20., 22., 23., 32., 34., 36.,
        38., 39., 48., 50., 52., 54., 55., 56., 58., 60., 62., 63.].sliced(5, 5);
    auto correct2 = iota([8, 8]).slice;
    const auto ret2 = prolongation!(double, 2)(arr2, correct2.shape);
    assert(ret2 == correct2);
}

unittest
{
    import mir.ndslice : fuse;

    auto arr = [
        [0.70986027, 0.05107005, 0.36803441, 0.91042483],
        [0.18354898, 0.5568611, 0.94596048, 0.99127882],
        [0.63025087, 0.33234683, 0.65401546, 0.98237209],
        [0.66271802, 0.48028311, 0.79653074, 0.18756112]
    ].fuse;
    auto correct6 = [
        [0.70986027, 0.38046516, 0.05107005, 0.20955223, 0.36803441, 0.91042483],
        [0.44670463, 0.3753351, 0.30396557, 0.48048151, 0.65699745, 0.95085182],
        [0.18354898, 0.37020504, 0.5568611, 0.75141079, 0.94596048, 0.99127882],
        [0.40689993, 0.42575195, 0.44460397, 0.62229597, 0.79998797, 0.98682545],
        [0.63025087, 0.48129885, 0.33234683, 0.49318115, 0.65401546, 0.98237209],
        [0.66271802, 0.57150057, 0.48028311, 0.63840692, 0.79653074, 0.18756112]
    ].fuse;
    auto correct7 = [
        [0.70986027, 0.38046516, 0.05107005, 0.20955223, 0.36803441, 0.63922962, 0.91042483],
        [0.44670463, 0.3753351, 0.30396557, 0.48048151, 0.65699745, 0.80392463, 0.95085182],
        [0.18354898, 0.37020504, 0.5568611, 0.75141079, 0.94596048, 0.96861965, 0.99127882],
        [0.40689993, 0.42575195, 0.44460397, 0.62229597, 0.79998797, 0.89340671, 0.98682545],
        [0.63025087, 0.48129885, 0.33234683, 0.49318115, 0.65401546, 0.81819377, 0.98237209],
        [0.64648445, 0.52639971, 0.40631497, 0.56579404, 0.7252731, 0.65511985, 0.5849666],
        [0.66271802, 0.57150057, 0.48028311, 0.63840692, 0.79653074, 0.49204593, 0.18756112]
    ].fuse;
    auto ret6 = prolongation!(double, 2)(arr, [6, 6]);
    auto ret7 = prolongation!(double, 2)(arr, [7, 7]);

    assert(approxEqual(ret6, correct6, 1e-2, 1e-8));
    assert(approxEqual(ret7, correct7, 1e-2, 1e-8));
}

unittest
{
    import std.range : generate;
    import std.random : uniform;
    import std.algorithm : fill;
    import mir.ndslice : strided;

    immutable size_t N = 4;
    auto A = slice!double(N, N);
    auto fun = generate!(() => uniform(0.0, 1.0));
    A.field.fill(fun);

    auto ret6 = prolongation!(double, 2)(A, [6, 6]);
    auto ret7 = prolongation!(double, 2)(A, [7, 7]);

    assert(ret6[0, 0 .. $].strided!(0)(2) == A[0, 0 .. $ - 1]);
    assert(ret6[0 .. $, 0].strided!(0)(2) == A[0 .. $ - 1, 0]);
    assert(ret6[$ - 2 .. $, 0 .. $].strided!(0, 1)(1, 2) == A[$ - 2 .. $, 0 .. $ - 1]);
    assert(ret6[0 .. $, $ - 2 .. $].strided!(0)(2) == A[0 .. $ - 1, $ - 2 .. $]);
    assert(ret6[$ - 2 .. $, $ - 2 .. $] == A[$ - 2 .. $, $ - 2 .. $]);

    assert(ret7[0, 0 .. $].strided!(0)(2) == A[0, 0 .. $]);
    assert(ret7[0 .. $, 0].strided!(0)(2) == A[0 .. $, 0]);
    assert(ret7[$ - 1 .. $, 0 .. $].strided!(1)(2) == A[$ - 1 .. $, 0 .. $]);
    assert(ret7[0 .. $, $ - 1 .. $].strided!(0)(2) == A[0 .. $, $ - 1 .. $]);
}

unittest
{
    import mir.ndslice : iota, sliced;
    import std.stdio : writeln;

    auto A = [
        0., 2., 4., 6.,
        14., 16., 18., 20.,
        28., 30., 32., 34.,
        42., 44., 46., 48.,
        98., 100., 102., 104.,
        112., 114., 116., 118.,
        126., 128., 130., 132.,
        140., 142., 144., 146.,
        196., 198., 200., 202.,
        210., 212., 214., 216.,
        224., 226., 228., 230.,
        238., 240., 242., 244.,
        294., 296., 298., 300.,
        308., 310., 312., 314.,
        322., 324., 326., 328.,
        336., 338., 340., 342.
    ].sliced(4, 4, 4);
    const auto correct = iota([7, 7, 7]).slice;
    const auto B = prolongation!(double, 3)(A, [7, 7, 7]);
    assert(correct == B);

}

unittest
{
    import mir.ndslice : iota, sliced;
    import std.stdio : writeln;

    auto A = [
        0., 2., 4., 5.,
        12., 14., 16., 17.,
        24., 26., 28., 29.,
        30., 32., 34., 35.,
        72., 74., 76., 77.,
        84., 86., 88., 89.,
        96., 98., 100., 101.,
        102., 104., 106., 107.,
        144., 146., 148., 149.,
        156., 158., 160., 161.,
        168., 170., 172., 173.,
        174., 176., 178., 179.,
        180., 182., 184., 185.,
        192., 194., 196., 197.,
        204., 206., 208., 209.,
        210., 212., 214., 215.
    ].sliced(4, 4, 4);
    const auto correct = iota([6, 6, 6]).slice;
    const auto B = prolongation!(double, 3)(A, [6, 6, 6]);
    assert(correct == B);

}
