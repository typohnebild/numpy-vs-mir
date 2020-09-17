#!/bin/sh

problempath=${1:-'../problems/'}

# source of virtual Python environment
run_virtual(){
	. ./venv/bin/activate || exit 1
	./run.sh "openblas" "$problempath"
	deactivate
}

# source of intel Python environment
run_intel(){
	. /tmp/intelpython3/bin/activate || exit 1
	./run.sh "intel" "$problempath"
	conda deactivate
}

run_intel
run_virtual
