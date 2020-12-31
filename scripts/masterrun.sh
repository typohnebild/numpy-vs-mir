#!/bin/sh

usage() {
	echo "Usage: $0 [-dio] [ -p problempath ] [ -b (multigrid|gsrb)]"
	exit 2
}

problempath='/tmp/problems/'
buildconf='multigrid'
# Flags that indicate with benchmark should be executed default all are true
RUN_INTEL=1
RUN_OPENBLAS=1
RUN_D=1
while getopts 'diop:b:' opts; do
	case $opts in
	d) RUN_D=0 ;;
	i) RUN_INTEL=0 ;;
	o) RUN_OPENBLAS=0 ;;
	p) problempath=$OPTARG ;;
	b) buildconf=$OPTARG ;;
	*) usage ;;
	esac
done

[ "$buildconf" = "multigrid" ] || [ "$buildconf" = "gsrb" ] || exit 1

# source of virtual Python environment
run_openblas() {
	cd ../Python/ || exit 1
	. ./venv/bin/activate || exit 1
	./run.sh "openblas" "$problempath" "${buildconf}"
	deactivate
}

# source of intel Python environment
run_intel() {
	cd ../Python/ || exit 1
	. ./intelpython3/bin/activate || exit 1
	./run.sh "intel" "$problempath" "${buildconf}"
	# conda deactivate
}

run_d() {
	./benchmark.sh "$problempath" "$1"
}

generate_problems.sh -p "$problempath" -b "$buildconf" -t "wave" || exit 1

oldpwd=$(pwd)

if [ $RUN_D -eq 1 ]; then
	cd ../D || exit 1
	dub build --force --compiler=ldc2 --build=release-nobounds --config="$buildconf"
	for x in "field" "naive" "slice" "ndslice"; do
		run_d "./$buildconf -s $x"
	done
fi

cd "$oldpwd" || exit 1

if [ $RUN_INTEL -eq 1 ]; then
	run_intel
	cd "$oldpwd" || exit 1
fi

if [ $RUN_OPENBLAS -eq 1 ]; then
	run_openblas
	cd "$oldpwd" || exit 1
fi
