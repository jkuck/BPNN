## Folder descriptions containing SAT problems
- sat_problems_noIndSets: we stripped independent sets and sampling sets from all SAT problems
- sat_problems_IndSetsRecomputed: we recomputed independent sets using MIS 
    (https://github.com/meelgroup/mis) with a timeout of 1000 seconds.  Only problems that finished
    within the timeout are in this directory

    
##Folder descriptions containg SAT counts/estimates
- exact_SAT_counts_noIndSets: we ran sharpSAT, dsharp, and F2 on the problems in 
    sat_problems_noIndSets (see folder SATestimates_F2_withIndSets for info with independent sets)
- sat_counts_uai: exact sat counts and lower bounds from our UAI paper, copied to this directory
- SATestimates_approxMC: estimates from approxMC run on problems in sat_problems_noIndSets
    and sat_problems_IndSetsRecomputed (when possible) we used the default confidence of 0.81 and epsilon=16
- SATestimates_approxMC_origData: estimates from approxMC run on their original data where
    problems contain precomputed independent sets and sampling sets
- SATestimates_F2_withIndSets: lower bound(s) from F2 with 3 and 6 one's per column
    run on sat_problems_IndSetsRecomputed
    
##Note
F2 (and possibly various other methods) do not fail gracefully when they timeout, so missing entries
correspond to timeouts

##Get comparison of BPNN runtimes and ApproxMC/F2 runtimes
- run learn_BP_SAT.py in test mode, saving results to json file
- run data/compare_BPNNvsHashing_runtimes.py
