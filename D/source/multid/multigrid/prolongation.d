module multid.multigrid.prolongation;

import mir.ndslice;

/++
This is the implementation of a prolongation
+/
void prolongation(T, size_t Dim)(Slice!(T*, Dim) e, auto fine_shape)
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