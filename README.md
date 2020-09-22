# Python vs. D using multigrid

## Abstract
TODO (ist das Kunst oder kann das weg?)

## Motivation

Python is a well known and often used programming language. Its C-based packages like numpy allow an efficient computation.
But there is another programming language comming up the last years **D**

D combines the best parts of C and Python and is therefore competitive to pythons numpy package. It serves an numpy like D-package called MIR.
This makes D comparabel to Python. Therefore, we decided to implement a multigrid implementation in both languages and compare their FLOPs.

There are already some comparison of D and Python, like [^fn1] and [^fn2] but compare relative simple instructions.
We wanted to compare these to with a more complex application from HPC and implemented a multigrid solver in both.

## Related Work

This [^fn1] and [^fn2].

## Methods
### Python multigrid
### D multigrid
### Meassurements

## Results

Benchmark W-cyclce, 2 pre-, postsmooth
Problemsize: 200, 400, 600 ... 4000

![](graphs/cip1e31709_flopss.png?raw=true)
![](graphs/cip1e31709_flops.png?raw=true)
![](graphs/cip1e31709_time.png?raw=true)

## Footnotes

[^fn1]: A Look at Chapel, D, and Julia Using Kernel Matrix Calculations [link](https://dlang.org/blog/2020/06/03/a-look-at-chapel-d-and-julia-using-kernel-matrix-calculations/)
[^fn2]: Mir Benchmark [link](https://github.com/tastyminerals/mir_benchmarks)
