#!/bin/sh

usage() {
	echo "Usage: $0 [ -p problempath ] [ -b (multigrid|gsrb)] [-t problemtype(wave|heat)]"
	exit 2
}

problempath='../problems'
buildconf='multigrid'
typ='wave'
while getopts 'p:b:t:' opts; do
	case $opts in
	p) problempath=$OPTARG ;;
	b) buildconf=$OPTARG ;;
	t) typ=$OPTARG ;;
	*) usage ;;
	esac
done

# sanitycheck
[ -z "$problempath" ] && usage
[ -z "$buildconf" ] && usage
[ -z "$problempath" ] && usage

generate_problem() {
	../Python/problemgenerator/generate.py "$problempath" 2 "$1" -t "$typ"
}

generate() {
	N=${1} # start
	STEP=${2}
	COUNT=${3}

	for _ in $(seq "$COUNT"); do
		generate_problem "$N"
		N=$((N + STEP))
	done
}

[ -e "$problempath" ] || mkdir -p "$problempath"

# delete existing problems
rm -f "$problempath/"*.npy

case $buildconf in
"multigrid")
	generate 16 16 3
	generate 64 64 20
	generate 1280 128 10
	generate 2560 256 6
	;;
"gsrb")
	generate 16 16 20
	generate 384 64 15
	;;

*) echo "$buildconf is not a supported buildconf" ;;
esac
