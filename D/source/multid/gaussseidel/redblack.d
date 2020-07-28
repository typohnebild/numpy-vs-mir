module multid.gaussseidel.redblack;

import std.stdio;
import mir.ndslice;

enum color {red = 0, black = 1};

/++
This is a Gauss Seidel Red Black implementation for 1D
+/
Slice!(T*, Dim) GS_RB(
    T,
    size_t Dim,
    size_t max_iter = 10_000_000,
    size_t norm_iter = 10_000,
    T eps = 1e-8)
    (Slice!(T*, Dim) F, Slice!(T*, Dim) U, double h)
{
    static assert(Dim >=1 && Dim <= 3);

    double h2 = h * h;

    // if (Dim == 1)
    // {
    //     // TODO
    // }
    // else if (Dim == 2)
    // {
    //     // TODO
    // }
    // else if (Dim == 3)
    // {
    //     // TODO
    // }

    foreach(it; 0..max_iter)
    {
        if(it % norm_iter == 0)
        {
            //TODO: implemenent apply_poisson
            // r = F - apply_poisson(U, h)
            // ...
        }
        // rote Halbiteration
        sweep(Dim) (color.red, F, U, h2)
        // schwarze Halbiteration
        sweep(Dim) (color.black, F, U, h2)
    }

    return U;
}

/++
This is a sweep implementation for 1D
+/
void sweep(size_t Dim: 1) (int color, auto F, auto U, double h2)
{
    int err;
    size_t n = F.shape(err);
    assert(err == 0);



}

/++
This is a sweep implementation for 2D
+/
void sweep(size_t Dim: 2) (int color, auto F, auto U, double h2)
{

}

/++
This is a sweep implementation for 3D
+/
void sweep(size_t Dim: 3) (int color, auto F, auto U, double h2)
{

}

