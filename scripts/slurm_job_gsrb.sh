#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
## allocate nodes for 6 hours
#SBATCH --time=01:00:00
# job name
#SBATCH --job-name=gsrb_dlang
#SBATCH --constraint=hwperf
# # first non-empty non-comment line ends SBATCH options

#load required modules (compiler, MPI, ...)
module load python
# run
start_run() {
	masterrun.sh -i -o -p ../problems -b gsrb
}
srun start_run
