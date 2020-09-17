#!/bin/sh
OUTFILE="results/outfile_$(hostname -s)_$(date +%d%m)"

problempath=${1:-'../problems/'}
[ -d "$problempath" ] || exit 1

get_infos(){
    ../scripts/getinfos.sh
    echo "#size:dim:time:cycles:error:flops*:"
}

[ -e "$OUTFILE" ] || get_infos >> "$OUTFILE" || exit 1

benchmark(){
    perf=$1
    problem=$2
    delay=1000

    cmd="./multid-static -p $problem -d $delay"
    if [ "$perf" = true ]
    then
        cmd="perf stat -M GFLOPS -D $delay $cmd"
    fi

    x=$($cmd 2>&1 || exit 1)
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

for problem in "$problempath/"*.npy; do
    dim=$(echo "$problem" | awk -F'_' '{print $2}')
    N=$(echo "$problem" | awk -F'_' '{print $3}')
    N=${N%%\.npy}

    for _ in $(seq $reps); do
        x=$(benchmark $perf "$problem")
        printf "%b:%b:%b\n" "$N" "$dim" "$x" >> "${OUTFILE}"
    done
done
