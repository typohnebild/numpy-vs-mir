#!/bin/sh

# script to get the cpu infos and the numpy config
# if $1 is np then the numpy configuration is also printed

get_cpu_infos(){
    lscpu | grep -i 'model name' | awk -F: '{ print $2}' | sed -e 's/^\s*//g'
}

get_numpy_config(){
    python -c 'import numpy; print("numpy version:", numpy.__version__); numpy.show_config()'
}

echo "############ INFOS"
date
get_cpu_infos
[ "$1" = "np" ] && get_numpy_config
echo "############ END INFOS"

if [ "$2" = true ]
then
    echo "size:dim:time:cycles:error:scalar_single:scalar_double:128b_packed_double:128b_packed_single:256b_packed_double:256b_packed_single:empty"
else
    echo "size:dim:time:cycles:error:empty"
fi
