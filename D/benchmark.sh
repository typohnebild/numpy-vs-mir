#!/bin/sh

problempath=${1:-'../problems/'}
[ -d "$problempath" ] || exit 1
binary=${2:-'./multigrid -s ndslice'}
sweeptype=$(echo "$binary" | sed -r 's/.+ -s (field|naive|slice|ndslice).*/\1/')
buildtype=$(echo "$binary" | sed -r 's/.+(multigrid|gsrb) .+/\1/')
# sanitiy check at least aginst empty strings
[ -z "$buildtype" ] && exit 1
[ -z "$sweeptype" ] && exit 1

OUTFILE="results/outfile_$(hostname -s)_$(date +%d%m)_${sweeptype}_${buildtype}"
# checks if the perf is usabele to count flops with GFOPS group

echo "$OUTFILE"

benchmark() {
	perf=$1
	problem=$2
	delay=1000
	delayPerf=1000

	cmd="$binary $([ "$buildtype" = 'gsrb' ] && echo '-v') -p $problem -d $delay"
	if [ "$perf" = true ]; then
		cmd="perf stat -M GFLOPS -D $delayPerf $cmd"
	fi

	x=$($cmd 2>&1) || exit 1
	out=$(echo "$x" | head -n 2 | tr '\n' ':' | tr ' ' ':' | awk -F':' '{print $23 ":" $11 ":" $14 ":"}')
	if [ "$perf" = true ]; then
		flops=$(echo "$x" | tail -n +3 | grep -i 'fp' | awk '{ print $1}' | tr '\n' ':')
		out="$out$flops"
	fi

	printf "%s\n" "$out"

}

perf=$(../scripts/check_perf.sh)

get_infos() {
	../scripts/getinfos.sh "mir" "$perf"
}

[ -e "$OUTFILE" ] || get_infos >>"$OUTFILE" || exit 1

reps=5

for _ in $(seq $reps); do
	for problem in "$problempath/"*.npy; do
		dim=$(echo "$problem" | awk -F'_' '{print $2}')
		N=$(echo "$problem" | awk -F'_' '{print $3}')
		N=${N%%\.npy}

		x=$(benchmark "$perf" "$problem") && printf "%b:%b:%b\n" "$N" "$dim" "$x" >>"${OUTFILE}"
	done
done
