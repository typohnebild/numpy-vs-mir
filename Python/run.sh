#!/bin/sh
# Fist argument is the type of accelerator usally intel mkl or openblas
# second is the base dir for the problems
OUTFILE="results/outfile_$(hostname -s)_$(date +%d%m)_${1}"

problempath=${2:-'../problems/'}
[ -d "$problempath" ] || exit 1

get_infos(){
    ../scripts/getinfos.sh "np"
}

[ -e "${OUTFILE}_1_numba" ] || get_infos >> "${OUTFILE}_1_numba" || exit 1
[ -e "${OUTFILE}_8_numba" ] || get_infos >> "${OUTFILE}_8_numba" || exit 1
[ -e "${OUTFILE}_1_nonumba" ] || get_infos >> "${OUTFILE}_1_nonumba" || exit 1
[ -e "${OUTFILE}_8_nonumba" ] || get_infos >> "${OUTFILE}_8_nonumba" || exit 1

benchmark(){
    perf=$1
    threads=$2
    problem=$3
    numba=$4
    delay=1000

    export OPENBLAS_NUM_THREADS=$threads
    export MKL_NUM_THREADS=$threads
    export NUMEXPR_NUM_THREADS=$threads
    export VECLIB_MAXIMUM_THREADS=$threads
    export OMP_NUM_THREADS=$threads
    cmd="./measure.py -p $problem -d $delay $numba"
    if [ "$perf" = true ]
    then
        cmd="perf stat -M GFLOPS -D $delay $cmd"
    fi

    x=$($cmd 2>&1)
    out=$(echo "$x" | head -n 2 | tr '\n' ':' | tr ' ' ':' | awk -F':' '{print $12 ":" $5 ":" $8 ":"}')
    if [ "$perf" = true ]
    then
        flops=$(echo "$x" | tail -n +3 | grep -i 'fp' | awk '{ print $1}' | tr '\n' ':')
        out="$out$flops"
    fi

    printf "%s\n" "$out"

}

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
perf=false
if [ "$paranoid" -lt 3 ]  && perf list eventgroups | grep -q FLOPS
then
    perf=true
fi

for threads in 1 8; do
    for numba in "-n" " "; do

        for problem in "$problempath/"*.npy; do
            dim=$(echo "$problem" | awk -F'_' '{print $2}')
            N=$(echo "$problem" | awk -F'_' '{print $3}')
            N=${N%%\.npy}
            for _ in $(seq 5); do
                # x=$(perf stat -M GFLOPS ./measure.py $N "$numba" 2>&1 | grep -i 'fp\|elapsed' | awk '{ print $1}' | tr '\n' ':')
                EXTENSION=$([ "$numba" = " " ] && echo "nonumba" || echo "numba")
                x=$(benchmark $perf $threads "$problem" "$numba")
                printf "%b:%b:%b\n" "$N" "$dim" "$x" >> "${OUTFILE}_${threads}_${EXTENSION}"
            done
        done
    done
done

exit 0
