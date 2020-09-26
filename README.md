# Python vs. D using multigrid


##  Content
1. [Motivation](#motivation)
2. [Related Work](#related-work)
3. [Methods](#methods)
    1. [Multigrid](#multigrid)
    2. [Gauss Seidel RedBlack](#gauss-seidel-redblack)
    3. [Poisson](#poisson)
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
It serves a numpy like D-package called MIR, which makes D comparable to Python.

There are already some comparisons between D and other competitors, like [^fn1] and [^fn2] but they
compare relatively simple instructions.
We compare D with Python with a more complex application from HPC and implement a multigrid solver
in both languages. The measurement takes place by solving the poisson equation in 2D with our
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
We want to solve the poisson equation with our multigrids and measure the FLOPs/sec for various
2D problems. Therefore, we are using the following hardware and software configurations:
* **Hardware:**
    * Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz | CPU max MHz: 4700 | CPU min MHz: 800
    * RAM: 64GB | Speed: ??? MHz
    * Kernel: Linux cip1e3 4.19.144-1-cip-amd64 x86_64 (gcc version 8.3.0 (Debian 8.3.0-6))
* **Versions:**
    * *Python*
        * Python 3.7.3
        * Numpy 1.19.0
        * Numba 0.50.1
        * Intel Python Distribution 2020.2.902
            * Numpy 1.18.5
            * Numba 0.49.1
    * *D*
        * LDC 1.23
        * mir-algorithm 3.9.6
        * mir-random 2.2.14

### Multigrid
see [^fn7]. The idea of Multigrid is to split the problem in smaler problems.
These smaler problems are computed by a solver - here Gauss Seidel RedBlack - and interpolated back
to the original problem size. The original problem shall be a poisson equation.
### Gauss Seidel RedBlack
see [^fn3]. Gauss Seidel RedBlack does things ...
### Poisson
see [^fn4]. Since we want to solve a poisson equation, we should cover this problem type here...

The Poisson Equation is &Delta;u = f

The discrete version looks like this:
(&Delta;u)<sub>i,j</sub> = 1/(h<sup>2</sup>) (u<sub>i+1,j</sub> + u<sub>i - 1, j</sub> + u<sub>i, j+1</sub> + u<sub>i, j-1</sub> - 4* u<sub>i, j</sub> )

## Implementation
### Python multigrid
[^fn5] and [^fn6]
### D multigrid
We did the same things as in Python.

## Measurements
We measured some fancy stuff.

## Results
Benchmark W-cycle, 2 pre-, postsmooth steps
Problemsize: 100, 200, 300, 400, 600 ..., 2000, 2200, 1400 ... 4000

### D Benchmark

![](graphs/cip1e32109D_flops.png?raw=true)
![](graphs/cip1e32109D_time.png?raw=true)

### Python Benchmark

![](graphs/cip1e32109numba_flops.png?raw=true)
![](graphs/cip1e32109numba_time.png?raw=true)

![](graphs/cip1e32109nonumba_flops.png?raw=true)
![](graphs/cip1e32109nonumba_time.png?raw=true)

### Benchmarks combined

![](graphs/cip1e32109_flops.png?raw=true)
![](graphs/cip1e32109_time.png?raw=true)

![](graphs/cip1e32109_FLOPS_subplots.png?raw=true)
![](graphs/cip1e32109_time_subplots.png?raw=true)

## Summary

Everything was fine :smiley:

tbd

## Footnotes

[^fn1]: A Look at Chapel, D, and Julia Using Kernel Matrix Calculations [link](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
[^fn2]: Mir Benchmark [link](https://github.com/tastyminerals/mir_benchmarks)
[^fn3]: Optimierung des Red-Black-Gauss-Seidel-Verfahrens auf ausgewählten x86-Prozessoren [link](https://www10.cs.fau.de/publications/theses/2005/Stuermer_SA_2005.pdf)
[^fn4]: FINITE DIFFERENCE METHODS FOR POISSON EQUATION [link](https://www.math.uci.edu/~chenlong/226/FDM.pdf)
[^fn5]: Intel distribution for Python [link](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html)
[^fn6]: Numba [link](https://numba.pydata.org/)
[^fn7]: Multigrid Tutorial [link](https://www.math.ust.hk/~mawang/teaching/math532/mgtut.pdf)
