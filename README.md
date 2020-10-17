# NumPy vs. MIR using multigrid

**TLDR:**
Implemented a multigrid method in Python and in D and tried to compare them.
Pictures are [here](#results).

If you have suggestions for improvements, fell free to open an issue.

## Content

- [NumPy vs. MIR using multigrid](#numpy-vs-mir-using-multigrid)
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
  - [Results and Discussion](#results-and-discussion)
    - [Solver Benchmark](#solver-benchmark)
    - [D Benchmark](#d-benchmark)
    - [Python Benchmark](#python-benchmark)
    - [Benchmarks combined](#benchmarks-combined)
    - [Table Multigrid-Cycles](#table-multigrid-cycles)
  - [Summary](#summary)

## Motivation

Python is a well known and often used programming language. Its C-based package NumPy allows an
efficient computation for a wide variety of problems.
Although Python is popular and commonly used it is worth to look if there are
other languages and if they perform similar or even better than the established
Python + NumPy combination.

In this sense we want to compare it with the D programming language and
its library _[MIR](https://www.libmir.org/)_ and find out to which extend they are comparable and
how the differences in the performance are.

To do so we choose a more complex application from HPC and implement a multigrid solver
in both languages. The measurement takes place by solving the Poisson equation in 2D with our
solvers.
The animation below shows the result of these calculations.

<div align="center">
    <p align="center">
        <img src="graphs/heatmap.gif?raw=true">
    </p>
</div>

## Related Work

There are already some comparisons between D and other competitors, like
[this](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
blog entry from Dr. Chibisi Chima-Okereke which deals with the comparison of D, Chapel and Julia.
It aims at kernel matrix operations like dot products, exponents, Cauchy, Gaussian, Power and some more.

In [MIR Benchmark](https://github.com/tastyminerals/mir_benchmarks), D was compared to Python and
Julia with respect to simple numerical operations like dot product, multiplication and sorting.
Similar to our approach, MIR and NumPy was used in those implementations.

Both works compare relatively simple instructions. We compare a more complex application, and not
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
boundaries of the Matrices. (see [here](https://www.math.uci.edu/~chenlong/226/FDM.pdf))

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
for the cells where the sum of indices is odd (black). (see [here](https://www10.cs.fau.de/publications/theses/2005/Stuermer_SA_2005.pdf))

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
approximation significantly. (see [here](https://www.math.ust.hk/~mawang/teaching/math532/mgtut.pdf))

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

In Python, we used _[Numba](https://numba.pydata.org/)_ to speed up the sweep in
Gauss-Seidel-Red-Black. The sweep basically performs the update step. The sweep implementation uses
the _NumPy_ array slices. The efficiency differences of with and without Numba are considered in the
[Python-Benchmark](#python-benchmark).

In D we implemented three Gauss-Seidel-Red-Black sweep approaches.
For this purpose, we implemented three different approaches:

1. Slices: Python like. Uses D Slices and Strides for grouping (Red-Black).
2. Naive: one for-loop for each dimension. Matrix-Access via multi-dimensional Array.
3. Fields: one for-loop for each dimension. Matrix is flattened. Access via flattened index.

The [first one](D/source/multid/gaussseidel/sweep.d#L98)
is the approach to implement the Gauss-Seidel in a way, that it "looks" syntactical
like the [Python](Python/multipy/GaussSeidel/GaussSeidel_RB.py#L85) implementation.
But since the indexing operator of the MIR slices did not support striding,
it was needed to do with a extra function call.
The [second](D/source/multid/gaussseidel/sweep.d#L176),
the "naive" version is an implementation as it can be found in an textbook.
The [third](D/source/multid/gaussseidel/sweep.d#L16) one is the most
optimized version with accessing the underling D-array of the MIR slice directly.
In the end it looked like a C/C++ implementation would look like.

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

  | Model Name                              | CPU min | CPU max  | L1d cache | L1i cache | L2 cache | L3 cache |
  | :-------------------------------------- | :------ | :------- | :-------- | :-------- | :------- | :------- |
  | Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz | 800 MHz | 4700 MHz | 32K       | 32K       | 256K     | 12288K   |

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

As benchmarks we used problems in size of
16, 32, 48, 64, 128, 192, .. 1216, 1280, 1408, 1536, ...,
2432, 2560, 2816, ..., 3840, 4096.
And solved the with a Multigrid W-cycle with 2 pre- and postsmoothing steps and
stopped when the problem was solved up to an epsilon of 1e-3.
For each permutation of the setup option a run was done 3 times.

In the [Python-Benchmark](#python-benchmark) we distinguish measurements between
number of threads (1 or 8) and with or without Numba.
We also experimented with the
_[Intel Python Distribution](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html)_
to speed up our implementation. The _Intel Python Distribution_ is a
combination of many Python-packages like _Numba_ or _NumPy_ that is
optimized for Intel CPUs.

In the [D-Benchmark](#d-benchmark) we differentiate the measurements between
the sweep implementations _slice_, _naive_ and _field_.

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
_scalar\_single_, _scalar\_double_, _128b\_packed\_double_,
_128b\_packed\_single_, _256b\_packed\_double_, _256b\_packed\_single_
for different floating-point operations.
_Perf_ offers the Metric Group _GFLOPS_ for these which counts all this hardware
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

This is suitable for our kind and complexity of project, but for more advanced
projects it might be suitable to use tools like
[PAPI](http://icl.cs.utk.edu/papi/) or
[likwid](https://github.com/RRZE-HPC/likwid),
which allow a more fine grain measurement.
But it would be necessary to provide a interface, especially for D,
that it can be used in the benchmarks.

## Results and Discussion

The vertical doted lines in the follwing pictures indicate the size of the L1, L2 and L3 cache.
The place of the vertical line is calculated with &#8730;(Cachesize/8) since we are calculating
with 64-bit values and the arrays are N &times; N big.

The memory-bandwidth (horizontal dotted line) in the figures below is calculated with the value of the
Triad-Strem benchmark as is was mentioned [here](#hardwaresoftware-setup).
This value was multiplied by 4/3, since the Stream benchmark is not aware of the write allocated
(see [here](https://moodle.rrze.uni-erlangen.de/pluginfile.php/16786/mod_resource/content/1/09_06_04-2020-PTfS.pdf) on slide 20)
and needs to be corrected.
To get the FLOP/s this value is then divided by 16, since for every 32 MB that is written
there are 2 floating point operations.
(see [here](https://moodle.rrze.uni-erlangen.de/pluginfile.php/16786/mod_resource/content/1/09_06_04-2020-PTfS.pdf) on slide 21)

### Solver Benchmark

We also compared the performance of the solvers in the different versions. Since the multigrid
algorithm uses the solver only on relative small problems, we also used problems up to a size of
100x100.

|               Flop/s                |                Time                |
| :---------------------------------: | :--------------------------------: |
| ![](graphs/gsrb_flops.png?raw=true) | ![](graphs/gsrb_time.png?raw=true) |

Here is already apparent that the D version with using the fields is the fastest one. While the
Python implementation using the Intel Distribution without Numba is the slowest one.
Furthermore, there is no difference in the single- and the multithreaded runs visible.
This might be an effect of the relative small array size.
The steps downwards that are especially visible in the time plots
are caused by the number of iterations that are needed to reach the stop criteria.
For example, solving the 60 &times; 60 problem performs 5000 Gauss-Seidel iterations,
while 65 &times; 65 problem only needs 4000 iterations.
This effect occurs in all the recorded samples, so it is plausible that it is caused
by numerical peculiarities of the problem.

### D Benchmark

|                  Flop/s                   |                   Time                   |
| :---------------------------------------: | :--------------------------------------: |
| ![](graphs/multigridD_flops.png?raw=true) | ![](graphs/multigridD_time.png?raw=true) |

In the right figure we see the FLOP/s achieved during the benchmarks with the different
D implementations.
The _field_ version performs best, then follows close the _naive_ version.
The _slice_ version achieves the lowest FLOP/s, since it is the most time consuming version,
as it can be seen in the right figure.


### Python Benchmark

|                     Flop/s                      |                      Time                      |
| :---------------------------------------------: | :--------------------------------------------: |
|  ![](graphs/multigridnumba_flops.png?raw=true)  |  ![](graphs/multigridnumba_time.png?raw=true)  |
| ![](graphs/multigridnonumba_flops.png?raw=true) | ![](graphs/multigridnonumba_time.png?raw=true) |

We split up the figures in different groups, the upper two pictures show the curves for the
benchmarks that are accelerated with Numba, the lower ones are without Numba.
What stands out in all benchmarks, is that there is no big difference visible between
single and multithreaded versions. This might be an effect of the relative small array sizes.
We did not actively parallelized the code and only set the allowed threads through the
environment variables.
When Numba is used, there is no big difference between the Intel Python distribution, that uses the
Intel MKL and the "plain" Python version accelerated with Openblas.
In the runs where Numba was not used, the Intel version is outperformed by the Openblas version.
One aspect that possibly plays into is the relatively old NumPy version that is used in the
Intel Python distribution.
The stepwise time curve is caused by more cycles needed to reach the stop criteria for the
corresponding problem size (see [table](#table-multigrid-cycles) below).
This bigger jumps in the needed time are also visible in the ups and downs in the FLOP/s figures.

### Benchmarks combined

|                      Flop/s                       |                       Time                       |
| :-----------------------------------------------: | :----------------------------------------------: |
|     ![](graphs/multigrid_flops.png?raw=true)      |     ![](graphs/multigrid_time.png?raw=true)      |
| ![](graphs/multigrid_FLOPS_subplots.png?raw=true) | ![](graphs/multigrid_time_subplots.png?raw=true) |

**_<span style="color:red">TODO</span>_**

### Table Multigrid-Cycles

The following table contains some details about how many multigrid cycles and levels were performed
for the according problem sizes:

| Problem size     | 16  | 32  | 48  | 64  | 128 | 192 | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 768 | 832 | 896 | 960 | 1024 | 1088 | 1152 | 1216 | 1280 | 1408 | 1536 | 1664 | 1792 | 1920 | 2048 | 2176 | 2304 | 2432 | 2560 | 2816 | 3072 | 3328 | 3584 | 3840 | 4096 |
| :--------------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Multigird cycles | 17  | 19  | 21  | 22  | 25  | 26  | 27  | 28  | 29  | 30  | 30  | 31  | 31  | 32  | 32  | 32  | 33  | 33  |  33  |  33  |  34  |  34  |  34  |  34  |  35  |  35  |  36  |  36  |  36  |  36  |  37  |  37  |  37  |  37  |  38  |  38  |  39  |  39  |  39  |
| # levels         |  4  |  5  |  5  |  6  |  7  |  7  |  8  |  8  |  8  |  8  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  9  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  11  |  11  |  11  |  11  |  11  |  11  |  11  |  11  |  11  |  11  |  12  |

## Summary

From a performance perspective the MIR implementations are superior to the NumPy implementation.
The big difference is especially visible in [this figures](#benchmarks-combined).
For the biggest problem the fastes D version takes around 20 seconds,
while the Intel version without takes almost 1300 secondes.
Propably this is mainly caused by the overhead of the Python interpreter
and might be reduced by more optimization efforts.

From a programming perspective it was a bit easier to use NumPy then MIR.
This is partially caused that we are somehow biased with the experience we already had in
the use with Python and NumPy.
In contrast, we only got to know D and MIR during this project.
Furthermore, the resources, especially the available documentation, for NumPy is a more exhaustive
and helpful then the one for MIR.