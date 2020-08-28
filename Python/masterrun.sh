#!/bin/sh

N=${1:-2}

# source of virtual Python environment
run_virtual(){
	. /proj/ciptmp/bu49mebu/venv/bin/activate || exit 1
	./run.sh ${N} "virtual"
	deactivate
}

# source of intel Python environment
run_intel(){
	. /proj/ciptmp/bu49mebu/IntelDistributionPython/intelpython3/bin/activate || exit 1
	./run.sh ${N} "intel"
	# deactivate
}

run_intel
run_virtual
