#!/bin/sh
# Fist argument is the type of accelerator usally intel mkl or openblas
# second is the base dir for the problems
OUTFILE="results/outfile_$(hostname -s)_$(date +%d%m)_${1}"

problempath=${2:-'../problems/'}
TYPE=${3:-'multigrid'}

[ -d "$problempath" ] || exit 1
[ -e "./benchmark_${TYPE}.py" ] || exit 1

benchmark() {
	perf=$1
	threads=$2
	problem=$3
	numba=$4
	# Loading the interpreter takes round about 500ms,
	#   so the delay is acutally delayed by additionally 500ms.
	# We want to make sure that perf starts measuring when
	#   Python benchmark is in wait state.
	delay=2500
	delayPerf=2400

	export OPENBLAS_NUM_THREADS=$threads
	export MKL_NUM_THREADS=$threads
	export NUMEXPR_NUM_THREADS=$threads
	export VECLIB_MAXIMUM_THREADS=$threads
	export OMP_NUM_THREADS=$threads
	export NUMBA_NUM_THREADS=$threads
	cmd="./benchmark_${TYPE}.py $([ "$TYPE" = "gsrb" ] && echo "-v") -p $problem -d $delay $numba"
	if [ "$perf" = true ]; then
		cmd="perf stat -M GFLOPS -D $delayPerf $cmd"
	fi

	# if it does not work on the first time try it asecond time but then leave
	x=$($cmd -t "$(date +%s%N)" 2>&1) || x=$($cmd -t "$(date +%s%N)" 2>&1) || { echo "$x" >>"./error.log" && exit 1; }

	out=$(echo "$x" | head -n 2 | tr '\n' ':' | tr ' ' ':' | awk -F':' '{print $12 ":" $5 ":" $8 ":"}')

	if [ "$perf" = true ]; then
		flops=$(echo "$x" | tail -n +3 | grep -i 'fp' | awk '{ print $1}' | tr '\n' ':')
		out="$out$flops"
	fi

	printf "%s\n" "$out"

}

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
perf=false
reps=3
if [ "$paranoid" -lt 3 ] && perf list eventgroups | grep -q FLOPS; then
	perf=true
fi

get_infos() {
	../scripts/getinfos.sh "np" "$perf"
}

[ -e "${OUTFILE}_1_numba_${TYPE}" ] || get_infos >>"${OUTFILE}_1_numba_${TYPE}" || exit 1
[ -e "${OUTFILE}_8_numba_${TYPE}" ] || get_infos >>"${OUTFILE}_8_numba_${TYPE}" || exit 1
[ -e "${OUTFILE}_1_nonumba_${TYPE}" ] || get_infos >>"${OUTFILE}_1_nonumba_${TYPE}" || exit 1
[ -e "${OUTFILE}_8_nonumba_${TYPE}" ] || get_infos >>"${OUTFILE}_8_nonumba_${TYPE}" || exit 1

for problem in "$problempath/"*.npy; do
	dim=$(echo "$problem" | awk -F'_' '{print $2}')
	N=$(echo "$problem" | awk -F'_' '{print $3}')
	N=${N%%\.npy}

	echo "$problem"

	for threads in 1 8; do
		for numba in "-n" " "; do
			for _ in $(seq $reps); do
				EXTENSION=$([ "$numba" = " " ] && echo "nonumba" || echo "numba")
				x=$(benchmark $perf $threads "$problem" "$numba") || continue
				printf "%b:%b:%b\n" "$N" "$dim" "$x" >>"${OUTFILE}_${threads}_${EXTENSION}_${TYPE}"
			done
		done
	done
done

exit 0
