module multid.multigrid.restriction;

import mir.ndslice;

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