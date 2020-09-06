module multid.multigrid.cycle;

import mir.ndslice : Slice, slice;
import std.exception : enforce;
import std.math : log2;
import std.conv : to;
import multid.multigrid.prolongation : prolongation;

class Cycle(T, size_t Dim)
{
protected:
    uint mu, l;
    Slice!(T*, Dim) F;
    T eps = 1e-30;
    T h;

    abstract Slice!(T*, Dim) presmooth(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract Slice!(T*, Dim) postsmooth(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract Slice!(T*, Dim) compute_residual(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract Slice!(T*, Dim) solve(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract Slice!(T*, Dim) restriction(Slice!(T*, Dim) U);

    Slice!(T*, Dim) compute_correction(Slice!(T*, Dim) r, uint l, T current_h)
    {
        auto e = slice!T(r.shape, 0);
        foreach (_; 0 .. mu)
        {
            e = do_cycle(r, e, l, current_h);

        }
        return e;
    }

    /++ adds the correction vector to the U +/
    Slice!(T*, Dim) add_correction(Slice!(T*, Dim) U, Slice!(T*, Dim) e)
    {

        U.field[] += e.field[];
        return U;
    }

    Slice!(T*, Dim) do_cycle(Slice!(T*, Dim) F, Slice!(T*, Dim) U, uint l, T current_h)
    {
        if (l <= 0 || U.shape[0] <= 1)
        {
            return solve(F, U, current_h);
        }

        U = presmooth(F, U, current_h);

        auto r = compute_residual(F, U, current_h * 2);

        r = restriction(r);

        auto e = compute_correction(r, l - 1, current_h * 2);

        e = prolongation!(T, Dim)(e, U.shape);
        U = add_correction(U, e);

        return postsmooth(F, U, current_h);
    }

public:
    /++
       Constructor for Cycle
    +/
    this(Slice!(T*, Dim) F, uint mu, uint l, T h)
    {
        enforce(l == 0 || log2(F.shape[0]) > l, "l is to big for F");
        this.F = F;
        this.l = l;
        this.h = h ? h : 1.0 / F.shape[0];
        this.mu = mu;
        if (this.l == 0)
        {
            this.l = F.shape[0].log2.to!uint - 1;
        }
    }

    /++
        This computes the residual
    +/
    Slice!(T*, Dim) residual(Slice!(T*, Dim) F, Slice!(T*, Dim) U)
    {
        return compute_residual(F, U, this.h);
    }

    /++
        The actual function to caculate a cycle
    +/
    Slice!(T*, Dim) cycle(Slice!(T*, Dim) U)
    {
        return do_cycle(this.F, U, this.l, this.h);
    }

    /++ Computes the l2 norm of U and the inital F+/
    abstract T norm(Slice!(T*, Dim) U);
}

class PoissonCycle(T, size_t Dim, uint v1, uint v2) : Cycle!(T, Dim)
{
    import multid.gaussseidel.redblack : GS_RB;

protected:
    override Slice!(T*, Dim) presmooth(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {

        return GS_RB!(T, Dim, v1)(F, U, current_h);
    }

    override Slice!(T*, Dim) postsmooth(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {

        return GS_RB!(T, Dim, v2)(F, U, current_h);
    }

    override Slice!(T*, Dim) compute_residual(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {
        import ap = multid.tools.apply_poisson;

        return ap.compute_residual!(T, Dim)(F, U, current_h);
    }

    override Slice!(T*, Dim) solve(Slice!(T*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {
        return GS_RB!(T, Dim, 100_000, 1_000, 1e-8)(F, U, current_h);
    }

    override Slice!(T*, Dim) restriction(Slice!(T*, Dim) U)
    {
        import multid.multigrid.restriction : weighted_restriction;

        return weighted_restriction!(T, Dim)(U);
    }

public:
    this(Slice!(T*, Dim) F, uint mu, uint l, T h)
    {
        super(F, mu, l, h);
    }

    override T norm(Slice!(T*, Dim) U)
    {
        import multid.tools.norm : nrmL2;

        auto res = residual(F, U);
        return nrmL2(res);
    }
}

unittest
{
    import std.range : generate;
    import std.random : uniform;
    import std.algorithm : fill, all;

    const size_t N = 10;

    auto U = slice!double([N, N], 1.0);
    U.field.fill(generate!(() => uniform(0.0, 1.0)));
    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    auto p = new PoissonCycle!(double, 2, 2, 2)(F, 2, 0, 0);
    p.cycle(U);

    assert(U[0][0 .. $].all!"a == 1.0");
    assert(U[1 .. $, 0].all!"a == 1.0");
    assert(U[$ - 1][1 .. $].all!"a== 0.0");
    assert(U[1 .. $, $ - 1].all!"a == 0.0");
}
