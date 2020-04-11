#!/bin/bash

for updates in 'SEQFIX' 'SEQRND' 'SEQMAX' 'PARALL'
do 
    for damping in 0.0 0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99 0.999
    do 
        for mflag in '--model_map_flag' ''
        do 
            for cflag in '--classification_flag' ''
            do
                python run_cpu.py python bp_map_marginal_spinglass.py $dflag $mflag --updates $updates --damping $damping
            done
        done
    done
done
