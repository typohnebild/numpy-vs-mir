module multid.tools.norm;
import mir.math.sum : sum;
import std.math : sqrt;
import mir.ndslice;

/++
    Computes the L2 norm
+/
auto nrmL2(V)(V v)
{

    return v.map!(x => x * x).sum.sqrt;
}
