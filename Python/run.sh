#!/bin/sh
OUTFILE="outfile_$(hostname)_$(date +%d%m)"

get_infos(){

    echo "############ INFOS"
    date
    lscpu | grep -i 'model name' | awk -F: '{ print $2}' | sed -e 's/^\s*//g'
    python -c 'import numpy; print("numpy version:", numpy.__version__); numpy.show_config()'
    echo "############ END INFOS"
}

[ -e "$OUTFILE" ] || get_infos >> "$OUTFILE" || exit 1


step=500
paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)

# intel=[true, false]

for numba in "True" "False" 
do
    N=500

    iter=${1:-2}
    if [ "$paranoid" -lt 3 ]  && perf list eventgroups | grep -q FLOPS
    then
        for _ in $(seq "$iter")
        do
            x=$(perf stat -M GFLOPS ./measure.py $N "$numba" 2>&1 | grep -i 'fp\|elapsed' | awk '{ print $1}' | tr '\n' ':')
            printf "%b:%b:%b\\n" "$numba" "$N" "$x" >> "$OUTFILE"
            N=$((N + step))
        done
    else
        for _ in $(seq "$iter")
        do
            x=$(/usr/bin/time -f %e ./measure.py $N "$numba" 2>&1)
            printf "%b:%b:%b\\n" "$numba" "$N" "$x" >> "$OUTFILE"
            N=$((N + step))
        done
    fi

done

exit 0
