#!/bin/sh

problempath=${1:-'../problems/'}
[ -d "$problempath" ] || exit 1
binary=${2:-'../multid -s field'}
sweeptype=$(echo "$binary" | sed -r 's/.+ -s (field|naive|slice) .*/\1/')
type=$(echo "$binary"| sed -r 's/.+(multid|gsrb) .+/\1/')

OUTFILE="results/outfile_$(hostname -s)_$(date +%d%m)_${sweeptype}_${type}"
echo $OUTFILE

benchmark(){
    perf=$1
    problem=$2
    delay=1000

    cmd="$binary -p $problem -d $delay"
    if [ "$perf" = true ]
    then
        cmd="perf stat -M GFLOPS -D $delay $cmd"
    fi

    x=$($cmd 2>&1) || exit 1
    out=$(echo "$x" | head -n 2 | tr '\n' ':' | tr ' ' ':' | awk -F':' '{print $23 ":" $11 ":" $14 ":"}')
    if [ "$perf" = true ]
    then
        flops=$(echo "$x" | tail -n +3 | grep -i 'fp' | awk '{ print $1}' | tr '\n' ':')
        out="$out$flops"
    fi

    printf "%s\n" "$out"

}

reps=3

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
perf=false
if [ "$paranoid" -lt 3 ]  && perf list eventgroups | grep -q FLOPS
then
    perf=true
fi

get_infos(){
    ../scripts/getinfos.sh "mir" "$perf"
}

[ -e "$OUTFILE" ] || get_infos $perf >> "$OUTFILE" || exit 1

for problem in "$problempath/"*.npy; do
    dim=$(echo "$problem" | awk -F'_' '{print $2}')
    N=$(echo "$problem" | awk -F'_' '{print $3}')
    N=${N%%\.npy}

    for _ in $(seq $reps); do
        x=$(benchmark $perf "$problem") || break
        printf "%b:%b:%b\n" "$N" "$dim" "$x" >> "${OUTFILE}"
    done
done
