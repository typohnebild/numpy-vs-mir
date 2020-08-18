#!/bin/sh
OUTFILE="results/outfile_$(hostname -s)_$(date +%d%m)_${2}"

get_infos(){

    echo "############ INFOS"
    date
    lscpu | grep -i 'model name' | awk -F: '{ print $2}' | sed -e 's/^\s*//g'
    python -c 'import numpy; print("numpy version:", numpy.__version__); numpy.show_config()'
    echo "############ END INFOS"
}

[ -e "${OUTFILE}_1_numba" ] || get_infos >> "${OUTFILE}_1_numba" || exit 1
[ -e "${OUTFILE}_8_numba" ] || get_infos >> "${OUTFILE}_8_numba" || exit 1
[ -e "${OUTFILE}_1_nonumba" ] || get_infos >> "${OUTFILE}_1_nonumba" || exit 1
[ -e "${OUTFILE}_8_nonumba" ] || get_infos >> "${OUTFILE}_8_nonumba" || exit 1


step=500
paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)

# intel=[true, false]

for threads in 1 8
do
    export OPENBLAS_NUM_THREADS=$threads
    export MKL_NUM_THREADS=$threads
    export NUMEXPR_NUM_THREADS=$threads
    export VECLIB_MAXIMUM_THREADS=$threads 
    export OMP_NUM_THREADS=$threads
    
    for numba in "True" "False" 
    do
        N=500
    
        iter=${1:-2}
        if [ "$paranoid" -lt 3 ]  && perf list eventgroups | grep -q FLOPS
        then
            for _ in $(seq "$iter")
            do
    	    for _ in $(seq 5)
    	    do
                    x=$(perf stat -M GFLOPS ./measure.py $N "$numba" 2>&1 | grep -i 'fp\|elapsed' | awk '{ print $1}' | tr '\n' ':')
    		EXTENSION=$([ $numba = "True" ] && echo "numba" || echo "nonumba")
                    printf "%b:%b\\n" "$N" "$x" >> "${OUTFILE}_${threads}_${EXTENSION}" 
    	    done
    	    N=$((N + step))
            done
        else
            for _ in $(seq "$iter")
            do
    	    for _ in $(seq 5)
     	    do
                    x=$(/usr/bin/time -f %e ./measure.py $N "$numba" 2>&1)
    		EXTENSION=$([ $numba = "True" ] && echo "numba" || echo "nonumba")
		printf "%b:%b\\n" "$N" "$x" >> "${OUTFILE}_${threads}_${EXTENSION}"
    	    done 
    	    N=$((N + step))
            done
        fi
    done
done

exit 0
