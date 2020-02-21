import os
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import operator
import random
from wandbCSV_to_latexTable import get_approxMC_estAndTime, get_F2_estAndTime, get_exact_ln_modelCount, get_exact_ln_modelCount
import json
from SAT_train_test_split import ALL_TRAIN_PROBLEMS

def make_SAT_estimate_figure(runtimes_dir='/atlas/u/jkuck/learn_BP/data/SAT_BPNN_runtimes/'):
    '''
    should be able to load dictionary of all training problem results for BPNN, from running learn_BP_SAT.py in testing mode
    results = {'squared_errors': squared_errors, 'runtimes': runtimes, 
               'BPNN_estimated_ln_counts': BPNN_estimated_counts, 
               'exact_ln_solution_counts': exact_solution_counts,
               'problem_names': problem_names}    
    '''
    
    with open(runtimes_dir + "trainSet_runtimesAndErrors_3layer.json", 'r') as json_file:
#     with open(runtimes_dir + "or50_trainSet_runtimesAndErrors_3layer.json", 'r') as json_file:
        results = json.load(json_file)

    print('sanity check, BPNN RMSE =', np.sqrt(np.mean(results['squared_errors'])))
        
    assert(len(results['runtimes']) == len(results['problem_names']))
    approxMC_ests = []
    exact_counts_approxMC = []
    
    F2_ests = []
    exact_counts_F2 = []
    
    BPNN_ests = []
    exact_counts_BPNN = []
    
    for idx, BPNN_est in enumerate(results['BPNN_estimated_ln_counts']):
        problem_name = results['problem_names'][idx]
        if not (problem_name in [problem['problem'] for problem in  ALL_TRAIN_PROBLEMS['problems_90']]):
            continue
        BPNN_ests.append(BPNN_est - results['exact_ln_solution_counts'][idx])
        exact_counts_BPNN.append(results['exact_ln_solution_counts'][idx])
        
        approxMC_lnZ_est, approxMC_time = get_approxMC_estAndTime(problem_name)
        if approxMC_lnZ_est is not None:
            approxMC_ests.append(approxMC_lnZ_est - results['exact_ln_solution_counts'][idx])
            exact_counts_approxMC.append(results['exact_ln_solution_counts'][idx])
        F2_lnZ_est, F2_varDeg3_time = get_F2_estAndTime(problem_name)
        if F2_lnZ_est is not None:
            F2_ests.append(F2_lnZ_est - results['exact_ln_solution_counts'][idx])        
            exact_counts_F2.append(results['exact_ln_solution_counts'][idx])
    
#     plt.plot(x_vals, perfect, '-', label='Zero Error')

    print("len(BPNN_ests):", len(BPNN_ests))
    print("len(exact_counts_approxMC):", len(exact_counts_approxMC))
    print("len(exact_counts_F2):", len(exact_counts_F2))

    
    plt.plot(exact_counts_BPNN, BPNN_ests, 'x', label='BPNN')
    plt.plot(exact_counts_approxMC, approxMC_ests, '+', label='ApproxMC')
    plt.plot(exact_counts_F2, F2_ests, '1', label='F2')


    # plt.xlabel('(f_max, c_max)', fontsize=14)
#     plt.xlabel(r'$\ln(\textrm{Exact Model Count})$', fontsize=14)
    plt.xlabel("ln(Exact Model Count)", fontsize=14)

    # plt.xlabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=14)

#     plt.ylabel(r'$\ln(\textrm{Estimate}) - \ln(\textrm{Exact Model Count})$', fontsize=15)
    plt.ylabel("ln(Estimate) - ln(Exact Model Count)", fontsize=15)

               
    plt.yscale('symlog')
    plt.xscale('log')


    plt.title('Exact vs. Estimated Model Counts', fontsize=20)
    # plt.legend(fontsize=12)    
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
            fancybox=True, ncol=3, fontsize=12, prop={'size': 12})
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})        

    plt.grid(True)

#     plt.xticks(np.arange(9), ['(.1, 5)', '(.1, 10)', '(.1, 50)', '(.2, 5)', '(.2, 10)', '(.2, 50)', '(1, 5)', '(1, 1)', '(1, 50)'])
    # plot_name = 'quick_plot.png'
    # plt.savefig(ROOT_DIR + 'sat_plots/' + plot_name) 
    plt.savefig('./exactVsEstModelCounts.eps', bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')   
    # plt.show()    
    

def get_BPNN_runtime_ratios(runtimes_dir='/atlas/u/jkuck/learn_BP/data/SAT_BPNN_runtimes/'):
    '''
    should be able to load dictionary of all training problem results for BPNN, from running learn_BP_SAT.py in testing mode
    results = {'squared_errors': squared_errors, 'runtimes': runtimes, 
               'BPNN_estimated_ln_counts': BPNN_estimated_counts, 
               'exact_ln_solution_counts': exact_solution_counts,
               'problem_names': problem_names}    
    '''
    with open(runtimes_dir + "trainSet_runtimesAndErrors_2layer.json", 'r') as json_file:
        results = json.load(json_file)

    print('sanity check, BPNN RMSE =', np.sqrt(np.mean(results['squared_errors'])))
        
    assert(len(results['runtimes']) == len(results['problem_names']))
    approxMC_over_BPNN_runtime_ratios = []
    F2_over_BPNN_runtime_ratios = []
    approxMC_over_F2_runtime_ratios = []
    
    BPNN_errors = []
    max_ratio=1
    for idx, problem_name in enumerate(results['problem_names']):
        BPNN_runtime = results['runtimes'][idx]
        approxMC_lnZ_est, approxMC_time = get_approxMC_estAndTime(problem_name)
        if approxMC_time is not None:
            if approxMC_time/BPNN_runtime > max_ratio:
                max_ratio = approxMC_time/BPNN_runtime
                print('new max_ratio=', max_ratio, "approxMC_time=", approxMC_time, "BPNN_runtime=", BPNN_runtime)
            approxMC_over_BPNN_runtime_ratios.append(approxMC_time/BPNN_runtime)
        F2_lnZ_est, F2_varDeg3_time = get_F2_estAndTime(problem_name)
        if F2_varDeg3_time is not None:
            F2_over_BPNN_runtime_ratios.append(F2_varDeg3_time/BPNN_runtime)        
        if (F2_varDeg3_time is not None) and (approxMC_time is not None):
            approxMC_over_F2_runtime_ratios.append(approxMC_time/F2_varDeg3_time)        


    print("total problem count =", len(results['problem_names']))
    print("fraction of problems completed by approxMC =", len(approxMC_over_BPNN_runtime_ratios)/len(results['problem_names']))
    print("fraction of problems completed by F2 =", len(F2_over_BPNN_runtime_ratios)/len(results['problem_names']))
    print()
    return approxMC_over_BPNN_runtime_ratios, F2_over_BPNN_runtime_ratios, approxMC_over_F2_runtime_ratios


def print_BPNN_runtime_ratio_stats():
    approxMC_over_BPNN_runtime_ratios, F2_over_BPNN_runtime_ratios, approxMC_over_F2_runtime_ratios = get_BPNN_runtime_ratios()
    print("np.min(approxMC_over_BPNN_runtime_ratios):", np.min(approxMC_over_BPNN_runtime_ratios))
    print("np.percentile(approxMC_over_BPNN_runtime_ratios, 10):", np.percentile(approxMC_over_BPNN_runtime_ratios, 10))        
    print("np.mean(approxMC_over_BPNN_runtime_ratios):", np.mean(approxMC_over_BPNN_runtime_ratios))
    print("np.median(approxMC_over_BPNN_runtime_ratios):", np.median(approxMC_over_BPNN_runtime_ratios))
    print("np.max(approxMC_over_BPNN_runtime_ratios):", np.max(approxMC_over_BPNN_runtime_ratios))
    print()
    print("np.min(approxMC_over_F2_runtime_ratios):", np.min(approxMC_over_F2_runtime_ratios))
    print("np.percentile(approxMC_over_F2_runtime_ratios, 10):", np.percentile(approxMC_over_F2_runtime_ratios, 10))    
    print("np.mean(approxMC_over_F2_runtime_ratios):", np.mean(approxMC_over_F2_runtime_ratios))
    print("np.median(approxMC_over_F2_runtime_ratios):", np.median(approxMC_over_F2_runtime_ratios))
    print("np.max(approxMC_over_F2_runtime_ratios):", np.max(approxMC_over_F2_runtime_ratios))    
    print()
    print("np.min(F2_over_BPNN_runtime_ratios):", np.min(F2_over_BPNN_runtime_ratios))
    print("np.percentile(F2_over_BPNN_runtime_ratios, 10):", np.percentile(F2_over_BPNN_runtime_ratios, 10))    
    print("np.mean(F2_over_BPNN_runtime_ratios):", np.mean(F2_over_BPNN_runtime_ratios))
    print("np.median(F2_over_BPNN_runtime_ratios):", np.median(F2_over_BPNN_runtime_ratios))
    print("np.max(F2_over_BPNN_runtime_ratios):", np.max(F2_over_BPNN_runtime_ratios))


if __name__ == "__main__":
    print_BPNN_runtime_ratio_stats()
#     make_SAT_estimate_figure()
