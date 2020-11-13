# NumPy vs. MIR using multigrid

**TLDR:**
Comparison of the implementations of a multigrid method in Python and in D.
Pictures are [here](#results-and-discussion).

If you have suggestions for improvements, feel free to open an issue or a pull request.

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
    - [Python Multigrid](#python-multigrid)
    - [D Multigrid](#d-multigrid)
    - [Differences in Red-Black Gauss–Seidel](#differences-in-red-black-gaussseidel)
  - [Measurements](#measurements)
    - [Hardware/Software Setup](#hardwaresoftware-setup)
    - [What was measured?](#what-was-measured)
    - [How was measured?](#how-was-measured)
  - [Results and Discussion](#results-and-discussion)
    - [Table Multigrid-Cycles](#table-multigrid-cycles)
    - [Solver Benchmark](#solver-benchmark)
    - [D Benchmark](#d-benchmark)
    - [Python Benchmark](#python-benchmark)
    - [Benchmarks combined](#benchmarks-combined)
  - [Summary](#summary)

## Motivation

Python is a well known and often used programming language. Its C-based package NumPy allows
efficient computation for a wide variety of problems.
Although Python is popular and commonly used it is worth to look if there are
other languages and if they perform similar or even better than the established
Python + NumPy combination.

In this sense we want to compare it with the D programming language and
its library _[MIR](https://www.libmir.org/)_ and find out to which extend they are comparable and
how the differences in the performance are.

To do so we choose a more complex application from HPC and implement a multigrid solver
in both languages.
The measurement is done by solving the Poisson equation in 2D with our
solvers.
The animation below shows the result of these calculations.

<div align="center">
<p align="center">
<figure align="center">
<img src="graphs/wave.gif?raw=true">
<br/>
<figcaption>Visualization of the results of the Multigrid algorithm after each cycle (lower left) and the L2-Norm of the residual (lower right).</figcaption>
</figure>
</p>
</div>

## Related Work

There are already some comparisons between D and other competitors, like
[this](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
blog entry from Dr. Chibisi Chima-Okereke which deals with the comparison of D, Chapel and Julia.
It aims at kernel matrix operations like dot products, exponents, Cauchy, Gaussian, Power and some more.

In [MIR Benchmark](https://github.com/tastyminerals/mir_benchmarks), D is compared to Python and
Julia with respect to simple numerical operations like dot product, multiplication and sorting.
Similar to our approach, MIR and NumPy is used in those implementations.

Both works compare the performance of individual matrix or vector operations.
We compare a more complex application, and not
just individual functions, by implementing a multigrid solver in D and Python
using MIR and NumPy.

## Methods

### Poisson Equation

The Poisson Equation is -&Delta;u = f and is used in various fields to describe processes like
fluid dynamics or heat distribution.
To solve it numerically, the finte-difference method is usually applied
for discretization.
The discrete version on rectangular 2D-Grid looks like this:

(&nabla;<sup>2</sup>u)<sub>i,j</sub> = <sup>1</sup>&frasl;<sub>(h<sup>2</sup>)</sub>
(u<sub>i+1,j</sub> + u<sub>i - 1, j</sub> +
u<sub>i, j+1</sub> + u<sub>i, j-1</sub> - 4 \* u<sub>i, j</sub> ) = f<sub>i,j</sub>

Where `h` is the distance between the grid points.
(see [here](https://www.math.uci.edu/~chenlong/226/FDM.pdf))

### Red-Black Gauss Seidel

The Gauss-Seidel method is a common iterative technique to solve systems of linear equations.

For `Ax = b` the element wise formula is this:

x<sup>(k+1)</sup><sub>i</sub> =
<sup>1</sup>&frasl;<sub>(a<sub>i,i</sub>)</sub>
(b<sub>i</sub> - &Sigma; <sub>i&lt;j</sub> a <sub>i,j</sub> x<sub>i,j</sub><sup>(k+1)</sup> - &Sigma;
<sub>i&gt;j</sub> a <sub>i,j</sub> x<sub>i,j</sub><sup>(k)</sup>)

The naive implementation is not good to parallelize since the computation is forced to be sequential
by construction.
This issue is tackled by grouping the grid points into two independent groups.
The corresponding method is called **Red-Black Gauss-Seidel**.
Therefore, the inside of the grid is divided into so-called red and black dots like a
chessboard.
It first calculates updates where the sum of indices is even (red), because these are
independent.
This step can be done in parallel.
Afterwards the same is done for the cells where the sum of indices is odd (black).
(see [here](https://www10.cs.fau.de/publications/theses/2005/Stuermer_SA_2005.pdf))

### Multigrid

A Multigrid method is an iterative solver for systems of equations in the form of `Ax = b`.
Where `A` is N &times; M matrix and `x` and `b` are Vectors with N entries.
The main idea of multigrid is to solve a relaxed version of the problem with less variables instead
of solving the problem directly.
The solution for `Ax = b` is approximated by using the residual `r = b - Ax`.
This residual describes the distance of the current `x` to the targeted solution and is used to
calculate the error `e`.
The error `e` is the solution to the system of equations `Ae = r`.
It is solved in a restricted version which means that the problem has been transformed to a
lower resolution.
The current approximation `x` is then corrected by adding the correction error `e`.
Since `e` has a lower resolution, it has to be interpolated to a higher resolution first.
The solving of `Ae = r` can be done recursively until the costs for solving
are negligible and can be done directly due to the smaller problem size.

The basic scheme of a multigrid cycle looks like the following:

- Pre-Smoothing – reducing initial errors calculating a few iterations of the Gauss–Seidel method.
- Residual Computation – computing residual error.
- Restriction – downsampling the residual to a lower resolution.
- Compute error – solve (recursively) the problem `Ae = r` on the restricted residual.
- Prolongation – interpolating the correction `e` back to the previous resolution.
- Correction – Adding prolongated correction error onto the solution approximation `x`.
- Post-Smoothing – reducing further errors using a few iterations of the Gauss–Seidel method.

Various cycle types can be defined by looping the computation of the correction error &mu; times.
Well known cycle types are _V-Cycle_ (&mu; = 1) and _W-Cycle_ (&mu; = 2).
Performing multiple multigrid cycles will reduce the error of the solution
approximation. (see [here](https://www.math.ust.hk/~mawang/teaching/math532/mgtut.pdf))

## Implementation

### Python Multigrid

The Python multigrid implementation is based on an abstract class _Cycle_.
It contains the basic logic of a multigrid cycle and how the correction shall be computed.

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

        r = self._compute_residual(F=F, U=U, h=h)

        r = self.restriction(r)

        e = self._compute_correction(r, l - 1, 2 * h)

        e = prolongation(e, U.shape)

        # correction
        U = U + e

        return self._postsmooth(F=F, U=U, h=h)
```

The class _PoissonCycle_ is a specialization of this abstract _Cycle_. Here, the class specific
methods like pre- and post-smoothing are implemented.
Both smoothing implementations and also the solver are using the Red-Black Gauss-Seidel algorithm.

### D Multigrid

The implementation in D is very similar to the implementation in Python. It is simply a translated
version from Python to D.

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
        if (l <= 1 || U.shape[0] <= 1)
        {
            return solve(F, U, current_h);
        }

        U = presmooth(F, U, current_h);

        auto r = compute_residual(F, U, current_h);

        r = restriction(r);

        auto e = compute_correction(r, l - 1, current_h * 2);

        e = prolongation!(T, Dim)(e, U.shape);
        U = add_correction(U, e);

        return postsmooth(F, U, current_h);
    }
```

### Differences in Red-Black Gauss–Seidel

The implementations of the multigrid only differ essentially in syntactical
matters. The main difference is in the used solver and smoother. More precisely, the difference
is the Gauss-Seidel method.

In Python, we use _[Numba](https://numba.pydata.org/)_ to speed up the sweep method in the
Red-Black Gauss-Seidel algorithm.
It basically performs the update step by using the _NumPy_ array slices.
The efficiency differences in using Numba or not are considered in the
[Python-Benchmark](#python-benchmark).

In order to estimate the fastest approach in D, we consider three variations of the
Red-Black Gauss-Seidel sweep:

1. Slices: Python like. Uses D Slices and Strides for grouping (Red-Black).
2. Naive: one for-loop for each dimension. Matrix-Access via multi-dimensional Array.
3. Fields: one for-loop. Matrix is flattened. Access via flattened index.

The [first one](D/source/multid/gaussseidel/sweep.d#L98)
is the approach to implement the Gauss-Seidel in a way, that it "looks" syntactical
like the [Python](Python/multipy/GaussSeidel/GaussSeidel_RB.py#L85) implementation.
But since the indexing operator of the MIR slices does not support striding,
it is needed to do with a extra function call of
[`strided`](http://mir-algorithm.libmir.org/mir_ndslice_dynamic.html#.strided).
The [second](D/source/multid/gaussseidel/sweep.d#L176),
the "naive" version is an implementation as it can be found in an textbook.
The [third](D/source/multid/gaussseidel/sweep.d#L16) one is the most
optimized version with accessing the underling D-array of the MIR slice directly.
In the end it looks like a C/C++ implementation would look like.

We do not compare these different variations in Python, because this would mean to use high level
Python for-loops which are not competitive for this application.
When using _Numba_ this might change a bit, because in some cases _Numba_ achieves more speedup on
python loops then on the _NumPy_ array operations.
When experimenting with that it showed up that the difference was in our use case not to big,
so we sticked with the sliced version.
Also because a loop version without _Numba_ would have been significant slower.

## Measurements

### Hardware/Software Setup

- **Software:**

  - _Python_
    - Python 3.7.3
    - NumPy 1.19.3
    - Numba 0.51.2
    - Intel Python Distribution 2020.4.912
      - NumPy 1.18.5
      - Numba 0.51.2
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

### What was measured?

As performance measures we use the execution time and the number of
floating-point operations (FLOP) per second (FLOP/s).

The example problem we choose is the [Poisson-Equation](#poisson-equation) with the two dimensional
function **_f(x,y) = sin(2&pi;x)cos(2&pi;y)_**.
The analytical solution is given by **_u(x,y) = sin(2&pi;x)cos(2&pi;y) / (8&pi;<sup>2</sup>)_**.
Therefore, it is very simple to check our results.
The boundaries are managed with the Dirichlet boundary condition, that means the boundaries of the
Matrices are not updated.
The iteration starts initially from zero.
Except from the boundary cells, there already initialized with the correct solution.
This [animation](#motivation) visualizes the results after each multigrid cycle.
In the lower left corner it shows the number of the visualized cycle,
as well as in the lower right corner the L2-Norm of the residual after each cycle.

As benchmarks for the multigrid implementations
([D Benchmark](#d-benchmark), [Python Benchmark](#python-benchmark),
[Benchmarks Combined](#benchmarks-combined)) we solve problems in size of
16, 32, 48, 64, 128, 192, .. 1216, 1280, 1408, 1536, ...,
2432, 2560, 2816, ..., 3840, 4096.
Each problem is solved with a Multigrid V-cycle with 2 pre- and postsmoothing steps.
As stop criteria we use an epsilon of 1e-6 multiplied with the squared problem size.
This makes the stop criteria independent of the problem size.
For each permutation of the setup option a run is done 3 times.
And in evaluation the median of them is considered.

We also compare the performance of the solvers in the different versions
in a [Solver Benchmark](#solver-benchmark).
Since the multigrid algorithm uses the solver only on relative small problems, we also use
problems up to a size of 1280 &times; 1280.
We generate 20 problems from size 16 &times; 16 to 320 &times; 320 by increasing the problem size
by 16 in each step.
From problem size 384 &times; 384 to 1280 &times; 1280 we increase the step size to 64.
The number of Gauss-Seidel iterations is fixed to 5000 for each problem size.

In order to make sure that each benchmark deals with the same problems, these are generated
beforehand and written into `.npy` files.
These files are later imported to the benchmark programs.

To see if parallelization causes some differences in performance, we measure the Python code with
1 and with 8 threads.
We do not actively parallelize the code and only set the number of allowed threads through the
environment variables (see [here](https://gitlab.cs.fau.de/bu49mebu/hpc-project/-/blob/master/Python/run.sh#L24)).
In addition, we also distinguish measurements between with or without optimizations using Numba.
As mentioned above, the sweep method in the Gauss-Seidel algorithm as well as the
intergrid operations restriction and prolongation are accelerated with the
[Numba jit decorator](https://numba.pydata.org/numba-doc/latest/reference/jit-compilation.html).
We also experiment with the
_[Intel Python Distribution](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html)_
to speed up our implementation.
The _Intel Python Distribution_ is a combination of many Python-packages like _Numba_ or _NumPy_
that is optimized for Intel CPUs.
These packages are accelerated with the _Intel MKL_.
When not using the _Intel Python Distribution_, NumPy is accelerated with the OpenBlas libary.

In the [D-Benchmark](#d-benchmark) we differentiate the measurements between
the sweep implementations _slice_, _naive_ and _field_ in the Gauss-Seidel method.

### How was measured?

To measure the execution time we use the `perf_counter()` from the
[Python time package](https://docs.python.org/3/library/time.html#time.perf_counter)
and in the D implementation the `Stopwatch` from the
[D standard library](https://dlang.org/phobos/std_datetime_stopwatch.html#.StopWatch)
is used.

To count the floating-point operations that occur while execution we use the
Linux tool [_perf_](https://perf.wiki.kernel.org/index.php/Main_Page), which is
build into the Linux kernel.
It allows to gather a enormous variety of performance counters, if they are implemented by the CPU.
The CPU we use offers the performance counters
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
To achieve this we use the delay option for _perf_, which delays the start of
the measurement, and also delay our programs accordingly.
The delay for the program is meant to be a bit longer than the actual startup
phase.
So the program needs to sleep after the warm-up until the delay is over.
It is meant that _perf_ starts to measure while the benchmark program is waiting till its delay
is over.
This is not problematic, because while waiting there should be no floating-point
operation that would spoil our results.
The actual execution time is measured separately on program side.

This procedure is sufficient for our kind and complexity of project, but for more advanced
projects it might be more suitable to use tools like
[PAPI](http://icl.cs.utk.edu/papi/) or
[likwid](https://github.com/RRZE-HPC/likwid),
which allow a more fine grain measurement.
But it would be necessary to provide an interface, especially for D,
so that it can be used in the benchmarks.

## Results and Discussion

The vertical doted lines in the follwing pictures indicate the size of the L1, L2 and L3 cache.
The place of the vertical line is calculated with &#8730;(Cachesize/8) since we are calculating
with 64-bit values and the arrays are N &times; N big.

The memory-bandwidth (horizontal dotted line) in the figures below is calculated with the value of the
Triad-Strem benchmark as it is mentioned [here](#hardwaresoftware-setup).
This value is multiplied by 4/3, since the Stream benchmark is not aware of the write allocated
(see [here](https://moodle.rrze.uni-erlangen.de/pluginfile.php/16786/mod_resource/content/1/09_06_04-2020-PTfS.pdf) on slide 20)
and needs to be corrected.
To get the FLOP/s this value is then divided by 16, since for every 32 MB that is written
there are 2 floating point operations.
(see [here](https://moodle.rrze.uni-erlangen.de/pluginfile.php/16786/mod_resource/content/1/09_06_04-2020-PTfS.pdf) on slide 21)

### Table Multigrid-Cycles

The following table contains some details about how many multigrid cycles and levels are performed
for the according problem sizes:

| Problem size     | 16  | 32  | 48  | 64  | 128 | 192 | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 768 | 832 | 896 | 960 | 1024 | 1088 | 1152 | 1216 | 1280 | 1408 | 1536 | 1664 | 1792 | 1920 | 2048 | 2176 | 2304 | 2432 | 2560 | 2816 | 3072 | 3328 | 3584 | 3840 | 4096 |
| :--------------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Multigird cycles |  4  |  7  |  8  |  8  |  9  | 10  | 10  | 10  | 11  | 11  | 11  | 11  | 11  | 11  | 11  | 11  | 11  | 11  |  11  |  12  |  11  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |  12  |
| # levels         |  3  |  4  |  4  |  5  |  6  |  6  |  7  |  7  |  7  |  7  |  8  |  8  |  8  |  8  |  8  |  8  |  8  |  8  |  9   |  9   |  9   |  9   |  9   |  9   |  9   |  9   |  9   |  9   |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  10  |  11  |

This table is representative for all benchmarks.
Since every variation of or Multigrid implementation does the same calculation the number of cycles
and levels are always the same. The number of levels are simply calculated with `⌊log2(N)⌋ - 1`.

### Solver Benchmark

|               Flop/s                |                Time                |
| :---------------------------------: | :--------------------------------: |
| ![](graphs/gsrb_flops.png?raw=true) | ![](graphs/gsrb_time.png?raw=true) |

Here is already apparent that the D version with using the fields is the fastest one.
While the Python implementation using the Intel Distribution without Numba is the slowest one.
Furthermore, there is no difference in the single- and the multithreaded runs visible.
This could be an effect of the relatively small array size, so multithreading would not be worthwhile.

Here it is noticeable that for all implementations - except Python without Numba - the FLOP/s increase
sharply before the L1 cache limit.
Between the L1 and L2 cache lines the graphs stop rising that much.
Up to the L3 cache line and beyond, the graphs start to decrease slightly and level out.
In contrast, Python implementations without Numba show no response to all cache limits.
This may be a consequence of the missing optimization without Numba.

All setups that are using Numba take almost the same time for the corresponding problem sizes.
So the FLOP/s graphs for these setups look verry similar and close
(see detailed view of [FLOP/s](graphs/gsrbnumba_flops.png) and [Time](graphs/gsrbnumba_time.png)).
Without Numba, the runs using the Intel environment are below the OpenBlas environment FLOP/s on
smaller problem sizes.
They soar and overtake OpenBlas between problem size 320 and 384, although the Intel's execution
time is even longer at this point.
This effect can only be caused by an abrupt increase of the FLOP by Intel.
Until problem size 576, the distance of Intel and OpenBlas is remarkably, then it shrinks
(see detailed view of [FLOP/s](graphs/gsrbnonumba_flops.png) and [Time](graphs/gsrbnonumba_time.png)).
Therefore, using the Intel environment without Numba is not appropriate.
However, both runs without Numba perform very poorly.
For big problems, the Python runs can be grouped by applying and not applying the Numba `jit`.

### D Benchmark

|                  Flop/s                   |                   Time                   |
| :---------------------------------------: | :--------------------------------------: |
| ![](graphs/multigridD_flops.png?raw=true) | ![](graphs/multigridD_time.png?raw=true) |

In the left figure we see the FLOP/s achieved during the benchmarks with the different
D implementations.
The _field_ version performs best, then follows close the _naive_ version.
The _slice_ version achieves the lowest FLOP/s, since it is the most time consuming version,
as it can be seen in the right figure.

The execution time of the _slice_ implementation has a higher slope for greater problems than
the _field_ and _naive_ implementations that remain closely. This results in a greater gap of FLOP/s
for greater problem sizes between the _slice_ the other two implementations.

Furthermore, the execution times for small problem sizes are in the range of several milliseconds.
As a result, even small changes in the execution time have a great impact to the
corresponding FLOP/s value.

### Python Benchmark

|                     Flop/s                      |                      Time                      |
| :---------------------------------------------: | :--------------------------------------------: |
|  ![](graphs/multigridnumba_flops.png?raw=true)  |  ![](graphs/multigridnumba_time.png?raw=true)  |
| ![](graphs/multigridnonumba_flops.png?raw=true) | ![](graphs/multigridnonumba_time.png?raw=true) |

We split up the figures in different groups, the upper two pictures show the curves for the
benchmarks that are accelerated with Numba, the lower ones are without Numba.
What stands out in all benchmarks just like in the [Solver Benchmark](#solver-benchmark), is that
there is no big difference visible between single and multithreaded versions.
This might be an effect of the small problem sizes in the lowest multigrid-level provided to the solver.
When Numba is used, there is no big difference between the Intel Python distribution, that uses the
Intel MKL and the "plain" Python version accelerated with Openblas.
In the runs where Numba is not used, the Intel version is slower than the Openblas version,
but runs more FLOP/s for problem sizes above 800.
One aspect that possibly plays into is the relatively old NumPy version that is used in the
Intel Python distribution.
The smaller steps in the time curve are caused by the increased multigrid-level for the
corresponding problem size (see [table](#table-multigrid-cycles)).
These jumps in the required execution time also influence the ups and downs of the FLOP/s values
accordingly.

### Benchmarks combined

|                      Flop/s                       |                       Time                       |
| :-----------------------------------------------: | :----------------------------------------------: |
|     ![](graphs/multigrid_flops.png?raw=true)      |     ![](graphs/multigrid_time.png?raw=true)      |
| ![](graphs/multigrid_FLOPS_subplots.png?raw=true) | ![](graphs/multigrid_time_subplots.png?raw=true) |

As already seen in the [Solver Benchmark](#solver-benchmark), the multigrid implementation in D
outperforms the Python implementations. Even the slowest D version (_slice_) is faster than
the fastest Python version. This may be due to the optimization level of the D compiler, but also to
the fact that compiled programs tend to be faster than interpreted ones.

More or less all implementations follow a similar pattern in behaviour of execution time and FLOP/s
as it can be seen in the figures with the single graphs.
For smaller problem sizes we can observe a sharp increase in FLOP/s until they reach a peak
round about problem size 500.
For bigger prolem sizes the FLOP/s slightly drop and finally level out.

## Summary

From a performance perspective the MIR implementations are superior to the NumPy implementation.
The big difference is especially visible in [this figures](#benchmarks-combined).
For the biggest multigird problem, the D versions take around 5-10 seconds,
while the Python versions take from 15 to 19 seconds.
Propably this is mainly caused by the overhead of the Python interpreter
and might be reduced by more optimization efforts.

From a programming perspective it was a bit easier to use NumPy than MIR.
This is partially caused that we are somehow biased with the experience we already had in
the use with Python and NumPy.
In contrast, we only got to know D and MIR during this project.
Furthermore, the resources, especially the available documentation, for NumPy is more exhaustive
and helpful then the one for MIR.
However, the D-community in the [D-Forum](https://forum.dlang.org/) is very helpful and we got
quick replies to our questions.

At the current point in time NumPy provides much more functionalities and utilities than MIR,
mainly due to its longer existence and big community.
But during our project, we did not miss essential features in MIR,
since the main data structure of MIR, the slices,
provides similar functionalities as the NumPy arrays.
The main difference, from a programmers point of view, lies in the indexing operator
and how striding is handled.

For convenience, there is also the library [_numir_](https://github.com/libmir/numir),
which provides some NumPy-like helper functions.
This allows a similar use of MIR compared to NumPy.

For those who are not afraid of statically typed programming languages and want to leave a lot of
optimization to a compiler, D in combination with MIR seems to be good choice for HPC applications.
