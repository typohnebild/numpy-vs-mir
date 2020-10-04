# Python vs. D using multigrid

**<span style="color:red">TODO Update TOC when finished</span>**

## Content

- [Python vs. D using multigrid](#python-vs-d-using-multigrid)
  - [Content](#content)
  - [Motivation](#motivation)
  - [Related Work](#related-work)
  - [Methods](#methods)
    - [Poisson Equation](#poisson-equation)
    - [Red-Black Gauss Seidel](#red-black-gauss-seidel)
    - [Multigrid](#multigrid)
  - [Implementation](#implementation)
    - [Multigrid](#multigrid-1)
      - [Python](#python)
      - [D](#d)
    - [Differences in Gauss–Seidel-Red-Black](#differences-in-gaussseidel-red-black)
  - [Measurements](#measurements)
    - [Hardware/Software Setup](#hardwaresoftware-setup)
    - [What was measured](#what-was-measured)
    - [How was measured](#how-was-measured)
  - [Results](#results)
    - [D Benchmark](#d-benchmark)
    - [Python Benchmark](#python-benchmark)
    - [Benchmarks combined](#benchmarks-combined)
  - [Summary](#summary)
  - [Footnotes](#footnotes)

## Motivation

Python is a well known and often used programming language. Its C-based package NumPy allows an
efficient computation for a wide variety of problems.
Although Python is popular and commonly used it is worth to look if there are
other languages and if they perform similar or even better than the established
Python + NumPy combination.

In this sense we want to compare it with the D programming language and
its library *MIR[^fn0]* and find out to which extend they are comparable and
how the differences in the performance are.

To do so we choose a more complex application from HPC and implement a multigrid solver
in both languages. The measurement takes place by solving the Poisson equation in 2D with our
solvers.
The animation below shows the result of these calculations.

![Animation](graphs/heatmap.gif?raw=true)

**<span style="color:red">It comes out that... tbd</span>**


## Related Work

There are already some comparisons between D and other competitors, like [^fn1] and [^fn2] but they
compare relatively simple instructions.

In reference [^fn2], D was compared to Python and Julia with respect to simple numerical operations
like dot product, multiplication and sorting. Similar to our approach, MIR and NumPy was used in
those implementations.

Reference [^fn1] deals with the comparison of D, Chapel and Julia. It aims kernel matrix operations
like dot products, exponents, Cauchy, Gaussian, Power and some more.

Based on the ideas both works, we compare a more complex application, and not
just individual functions, by implementing a multigrid solver in D and Python
using MIR and NumPy.

## Methods

### Poisson Equation

The Poisson Equation is -&Delta;u = f and has used in various fields to describe processes like
fluid dynamics or heatmaps. To solve it numerically, the finte-difference method is usually used for discretization.
The discrete version on rectangular 2D-Grid looks like this:

(&nabla;<sup>2</sup>u)<sub>i,j</sub> = <sup>1</sup>&frasl;<sub>(h<sup>2</sup>)</sub>
(u<sub>i+1,j</sub> + u<sub>i - 1, j</sub> +
u<sub>i, j+1</sub> + u<sub>i, j-1</sub> - 4 \* u<sub>i, j</sub> ) = f<sub>i,j</sub>

Where h is distance between the grid points.
The boundaries are managed by the Dirichlet boundary condition, since no update is performed on the
boundaries of the Matrices. (see [^fn4])

### Red-Black Gauss Seidel

The Gauss-Seidel method is a common iterative technique to solve systems of linear equations.

For Ax = b the element wise formula is this:

x<sup>(k+1)</sup><sub>i</sub> =
<sup>1</sup>&frasl;<sub>(a<sub>i,i</sub>)</sub>
(b<sub>i</sub> - &Sigma; <sub>i&lt;j</sub> a <sub>i,j</sub> x<sub>i,j</sub><sup>(k+1)</sup> - &Sigma;
 <sub>i&gt;j</sub> a <sub>i,j</sub> x<sub>i,j</sub><sup>(k)</sup>)

The naive implementation is not good to parallelize since the computation is forced to be sequential
by construction. This issue is tackled by grouping the grid points into two independent groups.
The corresponding method is called **Gauss-Seidel-Red-Black**.
Therefore, the inside of the grid is divided into so-called red and black dots like a
chessboard. First calculate updates where the sum of indices is even (red), because they are
independent. This step can be done in parallel. Afterwards the same is done
for the cells where the sum of indices is odd (black). (see [^fn3])

### Multigrid

The main idea of multigrid is to accelerate the convergence of a fine grid solution approximation
by recursively accelerating the convergence of a coarser grid solution approximation based on the
finer grid. This recursion is done until the costs for solving the grid is negligible.
Since the coarser grid is a representation of the finer grid, the error can be tracked back by
computing the prolongated residual in each recursion level.
During the backtracking, various cycle types can be defined by looping the correction &mu; times.
Well known cycle types are _V-Cycle_ (&mu; = 1) and _W-Cycle_ (&mu; = 2).

One multigrid cycle looks like the following:
- Pre-Smoothing – reducing high frequency errors using a few iterations of the Gauss–Seidel method.
- Residual Computation – computing residual error after the smoothing operation(s).
- Restriction – downsampling the residual error to a coarser grid.
- Prolongation – interpolating a correction computed on a coarser grid into a finer grid.
- Correction – Adding prolongated coarser grid solution onto the finer grid.
- Post-Smoothing – reducing further errors using a few iterations of the Gauss–Seidel method.

Performing multiple multigrid cycles will reduce the error of the solution
approximation significantly. (see [^fn7])


## Implementation

### Multigrid

#### Python

The Python multigrid implementation is based on an abstract class _Cycle_. It contains the basic
logic of a multigrid cycle and how the correction shall be computed.

```python
    def _compute_correction(self, r, l, h):
        e = np.zeros_like(r)
        for _ in range(self.mu):
            e = self.do_cycle(r, e, l, h)
        return e

    def do_cycle(self, F, U, l, h=None):
        if h is None:
            h = 1 / U.shape[0]

        if l <= 1 or U.shape[0] <= 1:
            return self._solve(F, U, h)

        U = self._presmooth(F=F, U=U, h=h)

        r = self._compute_residual(F=F, U=U, h=2 * h)

        r = self.restriction(r)

        e = self._compute_correction(r, l - 1, 2 * h)

        e = prolongation(e, U.shape)

        # correction
        U = U + e

        return self._postsmooth(F=F, U=U, h=h)
```
The class _PoissonCycle_ is a specialization of this abstract _Cycle_. Here, the class specific
methods like pre- and post-smoothing are implemented. Both smoothing implementations and also
the solver are based on Gauss-Seidel-Red-Black.

#### D

In D we did basically the same things as in Python.

```D
    Slice!(T*, Dim) compute_correction(Slice!(T*, Dim) r, uint l, T current_h)
    {
        auto e = slice!T(r.shape, 0);
        foreach (_; 0 .. mu)
        {
            e = do_cycle(r, e, l, current_h);

        }
        return e;
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
```

### Differences in Gauss–Seidel-Red-Black

The implementations of the multigrid only differ essentially in syntactical
matters. The main difference is in the used solver and smoother. More precisely, the difference
is the Gauss-Seidel method.

In Python, we used _Numba_[^fn6] to speed up the sweep in Gauss-Seidel-Red-Black. The sweep basically
performs the update step. The sweep implementation uses the _Numpy_ array slices.
The efficiency differences of with and without Numba are considered in the
[Python-Benchmark](#python-benchmark).

In D we implemented three Gauss-Seidel-Red-Black sweep approaches.
For this purpose, we implemented three different approaches:
1. Slices: Python like. Uses D Slices and Strides for grouping (Red-Black).
2. Naive: one for-loop for each dimension. Matrix-Access via multi-dimensional Array.
3. Fields: one for-loop for each dimension. Matrix is flattened. Access via flattened index.

## Measurements

### Hardware/Software Setup

- **Software:**
  - _Python_
    - Python 3.7.3
    - NumPy 1.19.0
    - Numba 0.50.1
    - Intel Python Distribution 2020.2.902
      - NumPy 1.18.5
      - Numba 0.49.1
  - _D_
    - LDC 1.23 [pre-built package](https://github.com/ldc-developers/ldc/releases)
    - mir-algorithm 3.9.6
    - mir-random 2.2.14

- **Hardware:**
  - Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz | CPU max MHz: 4700 | CPU min MHz: 800
  - | CPU-Cache | Size |
    |:---:|:---:|
    | L1d cache | 32K |
    | L1i cache | 32K |
    | L2 cache | 256K |
    | L3 cache | 12288K |
  - Kernel: Linux cip1e3 4.19.144-1-cip-amd64 x86_64 (gcc version 8.3.0 (Debian 8.3.0-6))
  - RAM: 64GB | Speed measured with [STREAM](https://github.com/jeffhammond/STREAM):
    ```
    -------------------------------------------------------------
    STREAM version $Revision: 5.10 $
    -------------------------------------------------------------
    This system uses 8 bytes per array element.
    -------------------------------------------------------------
    Array size = 10000000 (elements), Offset = 0 (elements)
    Memory per array = 76.3 MiB (= 0.1 GiB).
    Total memory required = 228.9 MiB (= 0.2 GiB).
    Each kernel will be executed 10 times.
    The *best* time for each kernel (excluding the first iteration)
    will be used to compute the reported bandwidth.
    -------------------------------------------------------------
    Number of Threads requested = 8
    Number of Threads counted = 8
    -------------------------------------------------------------
    Your clock granularity/precision appears to be 1 microseconds.
    Each test below will take on the order of 4697 microseconds.
      (= 4697 clock ticks)
    Increase the size of the arrays if this shows that
    you are not getting at least 20 clock ticks per test.
    -------------------------------------------------------------
    WARNING -- The above is only a rough guideline.
    For best results, please be sure you know the
    precision of your system timer.
    -------------------------------------------------------------
    Function    Best Rate MB/s  Avg time     Min time     Max time
    Copy:           22945.6     0.006989     0.006973     0.007011
    Scale:          22490.3     0.007176     0.007114     0.007224
    Add:            25086.2     0.009590     0.009567     0.009694
    Triad:          25104.3     0.009577     0.009560     0.009598
    -------------------------------------------------------------
    Solution Validates: avg error less than 1.000000e-13 on all three arrays
    -------------------------------------------------------------
    ```

### What was measured

As performance measures we used the execution time and the number of
floating-point operations (FLOP) per second (FLOP/s).

As benchmarks we used problems in size of 64, 128, 192, .. 1216, 1280, 1408, 1536, ..., 2432,
2560, 2816, ..., 3840, 4096.
And solved the with a Multigrid W-cycle with 2 pre- and postsmoothing steps and
stopped when the problem was solved up to an epsilon of 1e-3.
For each permutation of the setup option a run was done 3 times.

In the [Python-Benchmark](#python-benchmark) we distinguish measurements between number of
threads (1 or 8) and with or without Numba.
We also experimented with the _Intel Python Distribution_[^fn5] to speed up our implementation. The
_Intel Python Distribution_ is a combination of many Python-packages like _Numba_ or _Numpy_
that is optimized for Intel CPUs.

In the [D-Benchmark](#d-benchmark) we differentiate the measurements between the sweep implementations
_slice_, _naive_ and _field_.

### How was measured

To measure the execution time we used the `perf_counter()` from the
[Python time package](https://docs.python.org/3/library/time.html#time.perf_counter)
and in the D implementation the `Stopwatch` from the
[D standard library](https://dlang.org/phobos/std_datetime_stopwatch.html#.StopWatch)
was used.

To count the floating-point operations that occur while execution we used the
Linux tool [_perf_](https://perf.wiki.kernel.org/index.php/Main_Page), which is
build into the Linux kernel and allows to gather a enormous variety of
performance counters, if they are implemented by the CPU.
The CPU we used offered the performance counters
_scalar_single_, _scalar_double_, _128b_packed_double_, _128b_packed_single_,
_256b_packed_double_, _256b_packed_single_ for different floating-point operations.
_Perf_ offers the Metric Group *GFLOPS* for these which counts all this hardware
events.

Before starting the actual benchmark, there is the need for a startup
phase were the original problem is loaded and a small warm-up problem is solved.
This is especially crucial for the Python implementation when it is accelerated
with numba, since in this initialization phase the JIT-Compiler of numba is doing
his work.
So we want to avoid that _perf_ counts the FLOP/s that occur while this phase.
To achieve this we used the delay option for _perf_, which delays the start of
the measurement and also implemented a delay in our programs.
The delay for the program is meant to be a bit longer than the actual startup
phase. So the program needs to sleep after the warm-up until the delay is over.
This is especially needed in the Python implementation because the two delays are
not synchronized since it takes some time till the Python interpreter is
loaded and starts to run the program. So it is meant that _perf_ starts to
measure while the benchmark program is waiting till its delay is over.
This should be no problem, because while waiting there should be no floating-point
operation that would spoil our results. The time is measured separately
on program side.

This is suitable for our kind and complexity of project, but for more advanced projects it
might be suitable to use tools like [PAPI](http://icl.cs.utk.edu/papi/) or
[likwid](https://github.com/RRZE-HPC/likwid), which allow a more fine grain
measurement. But it would be necessary to provide a interface, especially for D,
that it can be used in the benchmarks.

## Results
***<span style="color:red">TODO: write some text for each sub-section</span>***

### D Benchmark

| Flop/s | Time |
|:---:|:---:|
|![](graphs/cip1e60210D_flops.png?raw=true)| ![](graphs/cip1e60210D_time.png?raw=true)|

### Python Benchmark

| Flop/s | Time |
|:---:|:---:|
|![](graphs/cip1e60210numba_flops.png?raw=true) | ![](graphs/cip1e60210numba_time.png?raw=true)|
|![](graphs/cip1e60210nonumba_flops.png?raw=true) | ![](graphs/cip1e60210nonumba_time.png?raw=true)|

### Benchmarks combined

| Flop/s | Time |
|:---:|:---:|
|![](graphs/cip1e60210_flops.png?raw=true) | ![](graphs/cip1e60210_time.png?raw=true)|

![](graphs/cip1e60210_FLOPS_subplots.png?raw=true)
![](graphs/cip1e60210_time_subplots.png?raw=true)

## Summary
***<span style="color:red">TODO</span>***

Compiled languages are faster?

Everything was fine :smiley:

tbd

## Footnotes

[^fn0]: Mir Software Library [link](https://www.libmir.org/)
[^fn1]: Chima-Okereke C., A Look at Chapel, D, and Julia Using Kernel Matrix Calculations [link](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
[^fn2]: Mir Benchmark [link](https://github.com/tastyminerals/mir_benchmarks)
[^fn3]: Optimierung des Red-Black-Gauss-Seidel-Verfahrens auf ausgewählten x86-Prozessoren [link](https://www10.cs.fau.de/publications/theses/2005/Stuermer_SA_2005.pdf)
[^fn4]: FINITE DIFFERENCE METHODS FOR POISSON EQUATION [link](https://www.math.uci.edu/~chenlong/226/FDM.pdf)
[^fn5]: Intel distribution for Python [link](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html)
[^fn6]: Numba [link](https://numba.pydata.org/)
[^fn7]: Multigrid Tutorial [link](https://www.math.ust.hk/~mawang/teaching/math532/mgtut.pdf)
