#!/bin/bash

for dflag in '--data_map_flag' ''
do 
    for mflag in '--model_map_flag' ''
    do 
        for updates in 'SEQFIX' 'SEQRND' 'SEQMAX' 'PARALL'
        do 
            for damping in 0.0 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.999
            do 
                python run_cpu.py python bp_map_spinglass.py $dflag $mflag --updates $updates --damping $damping
            done
        done
    done
done
