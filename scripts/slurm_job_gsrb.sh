#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
## allocate nodes for 4 hours
#SBATCH --time=04:00:00
# job name
#SBATCH --job-name=gsrb_dlang
#SBATCH --constraint=hwperf
# # first non-empty non-comment line ends SBATCH options

#load required modules (compiler, MPI, ...)
module load python
# run
srun ./masterrun.sh -i -o -p ~/problems -b gsrb
