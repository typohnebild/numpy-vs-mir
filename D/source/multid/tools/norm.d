module multid.tools.norm;

import mir.math : sqrt, fastmath;
import mir.ndslice : Slice, sliced;

/++
    Computes the L2 norm
+/
@fastmath
T nrmL2(T, size_t Dim)(Slice!(T*, Dim) v)
{
    T s = 0.0;
    foreach (x; v.field)
    {
        s += x * x;
    }
    return s.sqrt;
}

unittest
{
    assert([1, 2, 3, 4].sliced!double.nrmL2 == 30.0.sqrt);
    assert([1, 1].sliced!double.nrmL2 == 2.0.sqrt);
    assert([1, 1, 1, 1].sliced!double.nrmL2 == 2.0);
}
