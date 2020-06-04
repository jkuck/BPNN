## Folder descriptions containing SAT problems
- sat_problems_noIndSets: we stripped independent sets and sampling sets from all SAT problems
- sat_problems_IndSetsRecomputed: we recomputed independent sets using MIS 
    (https://github.com/meelgroup/mis) with a timeout of 1000 seconds.  Only problems that finished
    within the timeout are in this directory.  The problem blasted_case113 and 75-17-7-q were accidentally permuted
    (variable indices and variable orders within factors) from their original forms
- sat_problems_permuted: variable indices and the order of variables within clauses were randomly permuted.
    Problems were taken from sat_problems_IndSetsRecomputed when available, otherwise sat_problems_noIndSets.
    Only contain problems in ALL_TRAIN_PROBLEMS and ALL_TEST_PROBLEMS (data.SAT_train_test_split). 
    Computed using permute_SAT.py
    
## Folder descriptions containg SAT counts/estimates
- exact_SAT_counts_noIndSets: we ran sharpSAT, dsharp, and F2 on the problems in 
    sat_problems_noIndSets (see folder SATestimates_F2_withIndSets for info with independent sets)
- sat_counts_uai: exact sat counts and lower bounds from our UAI paper, copied to this directory
- SATestimates_approxMC: estimates from approxMC run on problems in sat_problems_noIndSets
    and sat_problems_IndSetsRecomputed (when possible) we used the default confidence of 0.81 and epsilon=16
- SATestimates_approxMC_origData: estimates from approxMC run on their original data where
    problems contain precomputed independent sets and sampling sets
- SATestimates_F2_withIndSets: lower bound(s) from F2 with 3 and 6 one's per column
    run on sat_problems_IndSetsRecomputed
- SATestimates_permutedProblems: estimates from approxMC and lower bound from F2 with 3 one's per column
    run on sat_problems_permuted
    
## Note
F2 (and possibly various other methods) do not fail gracefully when they timeout, so missing entries
correspond to timeouts

## Get comparison of BPNN runtimes and ApproxMC/F2 runtimes
- run learn_BP_SAT.py in test mode, saving results to json file
- run data/compare_BPNNvsHashing_runtimes.py
