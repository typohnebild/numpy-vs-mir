#!/bin/sh

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
perf=false
if [ -x "$(command -v perf)" ] && [ "$paranoid" -lt 3 ] && perf list eventgroups | grep -q FLOPS; then
	perf=true
fi
echo "$perf"
