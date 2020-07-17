#!/usr/bin/sh
OUTFILE="outfile_$(hostname)_$(date +%d%m)"
N=1000
for _ in $(seq 12)
do
    x=$(perf stat -M GFLOPS ./measure.py $N 2>&1 | grep -i 'fp\|elapsed' | awk '{ print $1}' | tr '\n' ':')
    printf "%b:%b\\n" "$N" "$x" >> "$OUTFILE"
    N=$((N + 1000))
done
