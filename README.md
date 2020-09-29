# Python vs. D using multigrid

## Content

1. [Motivation](#motivation)
2. [Related Work](#related-work)
3. [Methods](#methods)
   1. [Poisson](#poisson)
   2. [Red-Black Gauss Seidel](#gauss-seidel-redblack)
   3. [Multigrid](#multigrid)
4. [Implementation](#implementation)
   1. [Python](#python-multigrid)
   2. [D](#d-multigrid)
5. [Measurements](#measurements)
6. [Results](#results)
7. [Summary](#summary)
8. [Footnotes](#footnotes)

## Motivation

Python is a well known and often used programming language. Its C-based package numpy allows an
efficient computation for a wide variety of problems.

D combines the best parts of C and Python and is therefore competitive to pythons numpy package.
It serves a numpy like D-package called MIR [^fn0], which makes D comparable to Python.

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

see [^fn4]. Since we want to solve a Poisson equation, we should cover this problem type here...

The Poisson Equation is &Delta;u = f

The discrete version looks like this:

(&nabla;<sup>2</sup>u)<sub>i,j</sub> = <sup>1</sup>&frasl;<sub>(h<sup>2</sup>)</sub> (u<sub>i+1,j</sub> + u<sub>i - 1, j</sub> + u<sub>i, j+1</sub> + u<sub>i, j-1</sub> - 4 \* u<sub>i, j</sub> )

Where h is distance between the grid points.

### Red-Black Gauss Seidel

see [^fn3]. Red-Black Gauss Seidel does things ...
The Gauss-Seidel method is common iterative technique to solve systems of linear equations.

For Ax = b the element wise formula is this:

x<sup>k+1</sup><sub>i</sub> =
<sup>1</sup>&frasl;<sub>(a<sub>i,i</sub>)</sub>
(b<sub>i</sub> - &Sigma; <sub>i&lt;j</sub> a <sub>i,j</sub> x<sub>i,j</sub><sup>(k+1)</sup> - &Sigma; <sub>i&gt;j</sub> a <sub>i,j</sub> x<sub>i,j</sub><sup>(k)</sup>)

Not not good to parallelize => Red-Black version
First calculate updates where the sum of indices is even, because they are
independent and this step can be done in parallel. Afterwards the same is done
for the cells where the sum of indices is odd.

### Multigrid

see [^fn7]. The idea of Multigrid is to split the problem in smaller problems.
These smaller problems are computed by a solver - here Red-Black Gauss Seidel - and interpolated back
to the original problem size. The original problem shall be a Poisson equation.

## Implementation

### Python multigrid

[^fn5] and [^fn6]

### D multigrid

We did the same things as in Python.

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

### Nice meaningful Heading

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
_Perf_ offers for these the Metric Group *GFLOPS* which counts all this hardware
events.

Before we can start are our actual benchmark, there is the need for a startup
phase were the problem is loaded and small problem is solved. This is
especially crucial for the Python implementation when it is accelerated with
numba, since in this initialization phase the JIT-Compiler of numba is doing
his work.
So we want to avoid that _perf_ counts the FLOPs that occur while this phase.
To achieve this we used the delay option for _perf_, which delays the start of
the measurement and also implemented a delay in our programs.
The delay for the program is meant to be a bit longer then the actual startup
phase so the program needs to sleep for a while till the delay is over.
In Python implementation this is especially needed because the two delays are
not synchronized since the it takes some time till the Python interpreter is
loaded and starts to run the program. So it is meant that _perf_ starts to
measure while the benchmark program is waiting till its delay is over.
This should be no problem, because while waiting there should be no
floating-point
operation that would spoil our results and the time is measured separately.

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
