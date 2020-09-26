#!/bin/sh

problempath=${1:-'/tmp/problems/'}
buildconf=${2:-'multid'}

generate_problems(){
	for i in $(seq 1 20)
	do
		../Python/problemgenerator/generate.py "$problempath" 2 $((i*100))
	done

	N=2000
	for i in $(seq 1 10)
	do
		../Python/problemgenerator/generate.py "$problempath" 2 $((N +  i*200))
	done
}

# source of virtual Python environment
run_virtual(){
	cd ../Python/ || exit 1
	. ./venv/bin/activate || exit 1
	./run.sh "openblas" "$problempath"
	deactivate
}

# source of intel Python environment
run_intel(){
	cd ../Python/ || exit 1
	. /tmp/intelpython3/bin/activate || exit 1
	./run.sh "intel" "$problempath"
	conda deactivate
}

run_d(){
	./benchmark.sh "$problempath" "$1"
}

generate_problems

oldpwd=$(pwd)

cd ../D || exit 1
dub build --force --compiler=ldc --build=release-nobounds --config=${buildconf}
for x in "field" "naive" "slice"
do
	run_d "./multid -s $x"
done
cd "$oldpwd" || exit 1

run_intel
cd "$oldpwd" || exit 1

run_virtual
cd "$oldpwd" || exit 1
