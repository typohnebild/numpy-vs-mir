#!/bin/sh

problempath=${1:-'/tmp/problems/'}
buildconf=${2:-'multigrid'}

[ "$buildconf" = "multigrid" || "$buildconf" = "gsrb" ] || exit 1

generate_problems(){
	[ -e "$problempath" ] || mkdir -p "$problempath"
	# delete existing problems
	rm -f "$problempath/"*.npy
	STEP=$( [ "$buildconf" = "multigrid" ] && echo "100" || echo "5")
	# generate new problems
	for i in $(seq 1 20)
	do
		../Python/problemgenerator/generate.py "$problempath" 2 $((i*${STEP}))
	done
	if [ "$buildconf" = "multigrid" ]
	then
		N=2000
		for i in $(seq 1 10)
		do
			../Python/problemgenerator/generate.py "$problempath" 2 $((N +  i*200))
		done
	fi
}

# source of virtual Python environment
run_virtual(){
	cd ../Python/ || exit 1
	. ./venv/bin/activate || exit 1
	./run.sh "openblas" "$problempath" "${buildconf}"
	deactivate
}

# source of intel Python environment
run_intel(){
	cd ../Python/ || exit 1
	. ./intelpython3/bin/activate || exit 1
	./run.sh "intel" "$problempath" "${buildconf}"
	# conda deactivate
}

run_d(){
	./benchmark.sh "$problempath" "$1"
}

generate_problems

oldpwd=$(pwd)

cd ../D || exit 1
dub build --force --compiler=ldc2 --build=release-nobounds --config="$buildconf"
for x in "field" "naive" "slice"
do
	run_d "./$buildconf -s $x"
done
cd "$oldpwd" || exit 1

run_intel
cd "$oldpwd" || exit 1

run_virtual
cd "$oldpwd" || exit 1
