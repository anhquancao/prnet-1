#!/bin/zsh

# This script continously monitors your graphics cards memory utilization 
# and writes it to the file 'memory.txt'

echo '' > memory.txt
while true
do
    echo "$(date '+TIME:%H:%M:%S') $(nvidia-smi -q -i 0 -d MEMORY | sed '12!d'
    -)" >> memory.txt
    sleep 1
done