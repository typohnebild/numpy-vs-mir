# Python vs. D using multigrid

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
    - [Python multigrid](#python-multigrid)
    - [D multigrid](#d-multigrid)
  - [Measurements](#measurements)
    - [Hardware/Software Setup](#hardwaresoftware-setup)
    - [How was measured](#how-was-measured)
    - [What was measured](#what-was-measured)
  - [Results](#results)
    - [D Benchmark](#d-benchmark)
    - [Python Benchmark](#python-benchmark)
    - [Benchmarks combined](#benchmarks-combined)
  - [Summary](#summary)
  - [Footnotes](#footnotes)

## Motivation

Python is a well known and often used programming language. Its C-based package numpy allows an
efficient computation for a wide variety of problems.

D combines the efficency of C and the simplicity of Python and is therefore competitive to Pythons
Numpy package. It also serves a numpy like D-package called MIR [^fn0], which makes D comparable to
Python with respect to MIR and Numpy.

There are already some comparisons between D and other competitors, like [^fn1] and [^fn2] but they
compare relatively simple instructions.
We compare D with Python with a more complex application from HPC and implement a multigrid solver
in both languages. The measurement takes place by solving the Poisson equation in 2D with our
solvers.

![](graphs/heatmap.gif?raw=true)

It comes out that... tbd

## Related Work

In reference [^fn2], D was compared to Python and Julia with respect to simple numerical operations
like dot product, multiplication and sorting. Similar to our approach, MIR and Numpy was used in
those implementations.

Reference [^fn1] deals with the comparison of D, Chapel and Julia. It aims kernel matrix operations
like dot products, exponents, Cauchy, Gaussian, Power and some more.

Based on the ideas both works, we compare a more complex application by implementing a multigrid
solver in D and Python using MIR and Numpy.

## Methods

### Poisson Equation

Since we want to solve a Poisson equation, we should cover this problem type here...

The Poisson Equation is &Delta;u = f

The discrete version looks like this:

(&nabla;<sup>2</sup>u)<sub>i,j</sub> = <sup>1</sup>&frasl;<sub>(h<sup>2</sup>)</sub>
(u<sub>i+1,j</sub> + u<sub>i - 1, j</sub> +
u<sub>i, j+1</sub> + u<sub>i, j-1</sub> - 4 \* u<sub>i, j</sub> )

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

### Python multigrid

[^fn5] and [^fn6]

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
### D multigrid

We did the same things as in Python.

### Differences in the Red-Black Gauss–Seidel Implementation

The implementation of the multigrid differs essentially only in syntactical
matters. The main difference is in the used solver, though the Gauss-Seidel
methods. 
**TODO: More Bla on what is the difference and field slice stuff**

## Measurements

### Hardware/Software Setup

We want to solve the Poisson equation with our multigrids and measure the FLOPs/sec for various
2D problems. Therefore, we are using the following hardware and software configurations:

- **Hardware:**
  - Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz | CPU max MHz: 4700 | CPU min MHz: 800
  - RAM: 64GB | Speed: ??? MHz
  - Kernel: Linux cip1e3 4.19.144-1-cip-amd64 x86_64 (gcc version 8.3.0 (Debian 8.3.0-6))
- **Versions:**
  - _Python_
    - Python 3.7.3
    - Numpy 1.19.0
    - Numba 0.50.1
    - Intel Python Distribution 2020.2.902
      - Numpy 1.18.5
      - Numba 0.49.1
  - _D_
    - LDC 1.23
    - mir-algorithm 3.9.6
    - mir-random 2.2.14

### How was measured

As performance measures we used the execution time and the number of
floating-point operations (FLOP) per second (FLOP/s).
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

### What was measured

We compared the different implementations and setups on this benchmark.
We create problems in size of 64, 128, 192, .. 1216, 1280, 1408, 1536, ..., 2432,
2560, 2816, ..., 3840, 4096. And solved the with a multigrid W-cycle with 2
pre- and postsmoothing steps up to an epsilon of 1e-3. For each permutation of
the setup option a run was done 3 times.

## Results

Benchmark W-cycle, 2 pre-, postsmooth steps
Problemsize: 100, 200, 300, 400, 600 ..., 2000, 2200, 1400 ... 4000

### D Benchmark

| Flop/s | Time |
|:---:|:---:|
|![](graphs/cip1e32109D_flops.png?raw=true)| ![](graphs/cip1e32109D_time.png?raw=true)|

### Python Benchmark

| Flop/s | Time |
|:---:|:---:|
|![](graphs/cip1e32109numba_flops.png?raw=true) | ![](graphs/cip1e32109numba_time.png?raw=true)|
|![](graphs/cip1e32109nonumba_flops.png?raw=true) | ![](graphs/cip1e32109nonumba_time.png?raw=true)|

### Benchmarks combined

| Flop/s | Time |
|:---:|:---:|
|![](graphs/cip1e32109_flops.png?raw=true) | ![](graphs/cip1e32109_time.png?raw=true)|

![](graphs/cip1e32109_FLOPS_subplots.png?raw=true)
![](graphs/cip1e32109_time_subplots.png?raw=true)

## Summary

Everything was fine :smiley:

tbd

## Footnotes

[^fn0]: Mir Software Library [link](https://www.libmir.org/)
[^fn1]: A Look at Chapel, D, and Julia Using Kernel Matrix Calculations [link](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
[^fn2]: Mir Benchmark [link](https://github.com/tastyminerals/mir_benchmarks)
[^fn3]: Optimierung des Red-Black-Gauss-Seidel-Verfahrens auf ausgewählten x86-Prozessoren [link](https://www10.cs.fau.de/publications/theses/2005/Stuermer_SA_2005.pdf)
[^fn4]: FINITE DIFFERENCE METHODS FOR POISSON EQUATION [link](https://www.math.uci.edu/~chenlong/226/FDM.pdf)
[^fn5]: Intel distribution for Python [link](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html)
[^fn6]: Numba [link](https://numba.pydata.org/)
[^fn7]: Multigrid Tutorial [link](https://www.math.ust.hk/~mawang/teaching/math532/mgtut.pdf)
