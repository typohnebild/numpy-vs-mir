module multid.multigrid.cycle;

import mir.conv : to;
import mir.exception : enforce;
import mir.math : log2;
import mir.ndslice : Slice, slice, iota, sliced;
import multid.gaussseidel.redblack : SweepType;
import multid.multigrid.prolongation : prolongation;
import std.traits : isFloatingPoint;

/++
    Abstract base class for the Cycles it implements the base MG sheme
+/
class Cycle(T, size_t Dim) if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
{
protected:
    uint mu, l;
    Slice!(const(T)*, Dim) initialF;
    T[] Rdata;
    Slice!(T*, Dim + 1)[] temp;

    @nogc final auto R(size_t[Dim] shape)
    {
        return Rdata[0 .. shape.iota.elementCount].sliced(shape);
    }

    T h;

    abstract @nogc void presmooth(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract @nogc void postsmooth(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract @nogc Slice!(T*, Dim) compute_residual(Slice!(const(T)*, Dim) F, Slice!(const(T)*, Dim) U, T current_h);
    abstract @nogc void solve(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, T current_h);
    abstract @nogc void restriction(Slice!(T*, Dim) res, Slice!(const(T)*, Dim) U);

    @nogc void compute_correction(Slice!(T*, Dim) e, Slice!(const(T)*, Dim) r, uint d, T current_h)
    {
        e[] = 0;
        foreach (_; 0 .. mu)
        {
            do_cycle(r, e, d, current_h);
        }
    }

    /++ adds the correction vector to the U +/
    @nogc void add_correction(Slice!(T*, Dim) U, Slice!(const(T)*, Dim) e)
    {

        U.field[] += e.field[];
    }

    @nogc void do_cycle(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, uint d, T current_h)
    {
        if (d + 1 >= l || U.shape[0] <= 1)
        {
            solve(F, U, current_h);
            return;
        }

        presmooth(F, U, current_h);

        auto r = compute_residual(F, U, current_h);

        auto res = temp[d][0];
        restriction(res, r);

        auto cor = temp[d][1];
        compute_correction(cor, res, d + 1, current_h * 2);

        auto e = R(U.shape);
        prolongation(e, cor);

        add_correction(U, e);

        postsmooth(F, U, current_h);
    }

public:
    /++
       Constructor for Cycle
       Params:
        F = Dim-slice as righthandsite
        mu = indicator for type of cycle
        l = the depth of the multigrid cycle if it is set to 0, the maxmium depth is choosen
        h = is the distance between the grid points if set to 0 1 / F.length is used
    +/
    this(Slice!(const(T)*, Dim) F, uint mu, uint l, T h)
    {
        auto ls = F.shape[0].to!double.log2;
        enforce!"l is to big for F"(l == 0 || ls > l);
        this.initialF = F;
        this.Rdata = F.shape.slice!double.field;
        this.l = l;
        this.h = h != 0 ? h : 1.0 / F.shape[0];
        this.mu = mu;
        if (this.l == 0)
        {
            this.l = ls.to!uint - 1;
        }
        auto m = F.length;
        
        if (m > 1) do
        {
            m = m / 2 + 1;
            size_t[Dim + 1] shape = m;
            shape[0] = 2;
            temp ~= shape.slice!double;
            if (m == 2)
                break;
        }
        while(m > 2);
    }

    /++
        This computes the residual
    +/
    @nogc Slice!(T*, Dim) residual(Slice!(const(T)*, Dim) F, Slice!(const(T)*, Dim) U)
    {
        return compute_residual(F, U, this.h);
    }

    /++
        The actual function to caculate a cycle
    +/
    @nogc void cycle(Slice!(T*, Dim) U)
    {
        do_cycle(this.initialF, U, 0, this.h);
    }

    /++ Computes the l2 norm of U and the inital F+/
    @nogc abstract T norm(Slice!(const(T)*, Dim) U);
}

/++ Poisson Cycle:
    T = a floatingpoint datatype
    Dim = dimension {1,2,3}
    v1 = number of presmoothing steps
    v2 = number of postsmoothing steps
    eps = the epsilon the is used in the cycle esspecially in the solve step as stopcriteria
+/
final class PoissonCycle(T, size_t Dim, uint v1, uint v2, SweepType sweep = SweepType.field,
        T eps = 1e-8) : Cycle!(T, Dim) if (1 <= Dim && Dim <= 3 && isFloatingPoint!T)
{
    import multid.gaussseidel.redblack : GS_RB;

protected:
    override void presmooth(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {
        auto r = R(F.shape);
        T norm;
        auto it = GS_RB!(v1, 1_000, eps, sweep)(F, U, r, current_h, norm);
    }

    override void postsmooth(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {
        auto r = R(F.shape);
        T norm;
        auto it = GS_RB!(v2, 1_000, eps, sweep)(F, U, r, current_h, norm);
    }

    override Slice!(T*, Dim) compute_residual(Slice!(const(T)*, Dim) F, Slice!(const(T)*, Dim) U, T current_h)
    {
        import ap = multid.tools.apply_poisson;

        auto r = R(F.shape);
        ap.compute_residual!(T, Dim)(r, F, U, current_h);
        return r;
    }

    override void solve(Slice!(const(T)*, Dim) F, Slice!(T*, Dim) U, T current_h)
    {
        auto r = R(F.shape);
        T norm;
        auto it = GS_RB!(100_000, 5, eps, sweep)(F, U, r, current_h, norm);
    }

    override void restriction(Slice!(T*, Dim) res, Slice!(const(T)*, Dim) U)
    {
        import multid.multigrid.restriction : weighted_restriction;
        return weighted_restriction(res, U);
    }

public:
    /++
       Params:
        F = Dim-slice as righthandside
        mu = indicator for type of cycle
        l = the depth of the multigrid cycle if it is set to 0, the maxmium depth is choosen
        h = is the distance between the grid points if set to 0 1 / F.length is used
    +/
    this(Slice!(const(T)*, Dim) F, uint mu, uint l, T h)
    {
        super(F, mu, l, h);
    }

    override T norm(Slice!(const(T)*, Dim) U)
    {
        import multid.tools.norm : nrmL2;

        auto res = residual(initialF, U);
        return nrmL2(res);
    }
}

unittest
{
    import mir.algorithm.iteration : all;
    import multid.tools.util : randomMatrix;

    const size_t N = 10;
    immutable h = 1.0 / N;

    auto U = randomMatrix!(double, 2)(N);

    U[0][0 .. $] = 1.0;
    U[1 .. $, 0] = 1.0;
    U[$ - 1][1 .. $] = 0.0;
    U[1 .. $, $ - 1] = 0.0;

    auto F = slice!double([N, N], 0.0);
    import multid.tools.apply_poisson : compute_residual;
    import multid.tools.norm : nrmL2;

    const norm_before = compute_residual(F, U, h).nrmL2;
    F[0][0 .. $] = 1.0;
    F[1 .. $, 0] = 1.0;
    F[$ - 1][1 .. $] = 0.0;
    F[1 .. $, $ - 1] = 0.0;
    auto p = new PoissonCycle!(double, 2, 2, 2)(F, 2, 0, h);
    p.cycle(U);

    const norm_after = compute_residual(F, U, h).nrmL2;

    assert(U[0][0 .. $].all!"a == 1.0");
    assert(U[1 .. $, 0].all!"a == 1.0");
    assert(U[$ - 1][1 .. $].all!"a == 0.0");
    assert(U[1 .. $, $ - 1].all!"a == 0.0");

    // it should be at least a bit smaller than before
    assert(norm_after <= norm_before);
}
