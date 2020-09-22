# Python vs. D using multigrid

## Abstract
TODO (ist das Kunst oder kann das weg?)

## Motivation

Python is a well known and often used programming language. Its C-based packages like numpy allow an efficient computation.
But there is another programming language comming up the last years **D**

D combines the best parts of C and Python and is therefore competitive to pythons numpy package. It serves an numpy like D-package called MIR.
This makes D comparabel to Python. 

There are already some comparison of D and Python, like [^fn1] and [^fn2] but compare relative simple instructions.
We wanted to compare these to with a more complex application from HPC and implemented a multigrid solver in both.

## Related Work

This [^fn1] and [^fn2].

## Methods
We want to solve heat maps with multigrids.

### Multigrid
see [^fn7]
### Gauss Seidel RedBlack 
see [^fn3]
### Poisson 
see [^fn4]

## Implementation
### Python multigrid
[^fn5] and [^fn6]
### D multigrid
We did the same things as in Python.

## Meassurements
We meassured some fancy stuff.

## Results
Benchmark W-cyclce, 2 pre-, postsmooth
Problemsize: 200, 400, 600 ... 4000

### D Benchmark

![](graphs/cip1e32109_flops.png?raw=true)
![](graphs/cip1e32109_time.png?raw=true)
![](graphs/cip1e32109_FLOPS_subplots.png?raw=true)
![](graphs/cip1e32109_time_subplots.png?raw=true)

## Footnotes

[^fn1]: A Look at Chapel, D, and Julia Using Kernel Matrix Calculations [link](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
[^fn2]: Mir Benchmark [link](https://github.com/tastyminerals/mir_benchmarks)
[^fn3]: Optimierung des Red-Black-Gauss-Seidel-Verfahrens auf ausgewählten x86-Prozessoren [link](https://www10.cs.fau.de/publications/theses/2005/Stuermer_SA_2005.pdf)
[^fn4]: FINITE DIFFERENCE METHODS FOR POISSON EQUATION [link](https://www.math.uci.edu/~chenlong/226/FDM.pdf)
[^fn5]: Intel distribution for Python [link](https://software.intel.com/content/www/us/en/develop/tools/distribution-for-python.html)
[^fn6]: Numba [link](https://numba.pydata.org/)
[^fn7]: Multigrid Tutorial [link](https://www.math.ust.hk/~mawang/teaching/math532/mgtut.pdf)
