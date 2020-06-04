import csv
from SAT_train_test_split import ALL_TRAIN_PROBLEMS
import math
from decimal import Decimal
import os
import numpy as np

wandb_results_csv = 'wandb_sat_results/train_val_same_category.csv'
wandb_results_csv_finished = 'wandb_sat_results/train_val_same_category_finished.csv'



train_val_split_options = ['random_shuffle', 'easyTrain_hardVal']
MSG_PASSING_ITERS_options = ['2', '3', '5', '10']
PROBLEM_CATEGORY_TRAIN_options = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', 'problems_90']


def get_dict_of_results(wandb_results_csv):
    '''
    read in a csv file containing results from wandb and return a dictionary of the results
    '''
    dict_of_results = {}    
    with open(wandb_results_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
    #             continue
            print(row)
            dict_of_results[(row["PROBLEM_CATEGORY_TRAIN"], row["train_val_split"], row["MSG_PASSING_ITERS"])] = {"RMSE_training": float(row["RMSE_training"]), "RMSE_val":float(row["RMSE_val"])}
    #         print(dict_of_results)
    #         sleep(sdf)
            line_count += 1
    return dict_of_results
        
    
    
def get_approxMC_estAndTime(problem, approxMC_estimates_directory='/atlas/u/jkuck/learn_BP/data/SATestimates_approxMC/'):
    '''
    Get the estimate of ln(solution count) from approxMC and the time it required.  Time is
    the smaller of (approxMC run on the formula with no MIS) and (approxMC run on the formula
    with MIS + the time to compute the MIS)
    '''
    approxMC_solCountEstimate = None
    if os.path.isfile(approxMC_estimates_directory + problem + '.txt'):
        
        with open(approxMC_estimates_directory + problem + '.txt', 'r') as f_approxMC:
            for line in f_approxMC:
                if line.startswith('approxMC time_out:'):
                    if line.split()[2] == "True":
                        timeout_noIndSet = True
                    else:
                        timeout_noIndSet = False
                        assert(line.split()[2] == "False")
                    if not timeout_noIndSet and line.split()[6] != 'None':
                        approxMC_time = float(line.split()[6])
                        approxMC_solCountEstimate = float(line.split()[4])
                    elif not timeout_noIndSet:
                        print("approxMC did not timeout, but solution count is None??")
                if line.startswith('approxMC_runWithIndependentSet time_out:'):
                    if line.split()[2] == "True":
                        timeout_withIndSet = True
                    else:
                        timeout_withIndSet = False
                        assert(line.split()[2] == "False")                        
                    if not timeout_withIndSet:                        
                        approxMC_time_withIndSet = float(line.split()[6]) + float(line.split()[10])
                        approxMC_solCountEstimate_withIndSet = float(line.split()[4])
                        if approxMC_solCountEstimate is None:
                            approxMC_time = approxMC_time_withIndSet
                            approxMC_solCountEstimate = approxMC_solCountEstimate_withIndSet
                        elif approxMC_time_withIndSet < approxMC_time:
                            approxMC_time = approxMC_time_withIndSet
                            approxMC_solCountEstimate = approxMC_solCountEstimate_withIndSet
                            assert(np.abs(approxMC_solCountEstimate - approxMC_solCountEstimate_withIndSet) < 8), ("approxMC with and without independent sets differs by more than expected:", approxMC_solCountEstimate, approxMC_solCountEstimate_withIndSet)
#                             print("approxMC with independed set is faster than without")
                        else:
                            pass
#                             print("approxMC with independed set is SLOWER than without")

    if approxMC_solCountEstimate is not None:
        approxMC_lnZ_est = approxMC_solCountEstimate/math.log(math.e, 2)
    else:
        approxMC_lnZ_est = None
        approxMC_time = None
        
    return approxMC_lnZ_est, approxMC_time


def get_F2_estAndTime(problem, F2_estimates_noMIS_directory='/atlas/u/jkuck/learn_BP/data/exact_SAT_counts_noIndSets/',
                     F2_estimates_withMIS_directory='/atlas/u/jkuck/learn_BP/data/SATestimates_F2_withIndSets/',
                     sat_problems_withMIS_directory='/atlas/u/jkuck/learn_BP/data/sat_problems_IndSetsRecomputed/'):
    '''
    Get the estimate of ln(solution count) from F2 and the time it required.  Time is
    the smaller of (F2 run on the formula with no MIS) and (F2 run on the formula
    with MIS + the time to compute the MIS)
    '''
    if os.path.isfile(F2_estimates_noMIS_directory + problem + '.txt'):
        #F2 without MIS        
        F2_varDeg3_time = None
        F2_varDeg3_log2_lowerBound = None        
        with open(F2_estimates_noMIS_directory + problem + '.txt', 'r') as f_approxF2:
            for line in f_approxF2:
                if line.strip().split(" ")[0] == 'biregular_variable_degree_3_Tsol_1' and line.strip().split(" ")[2] == 'False': #not a timeout
                    F2_varDeg3_log2_lowerBound = float(line.strip().split(" ")[4])
                    F2_varDeg3_time = float(line.strip().split(" ")[6])

        #F2 with MIS
        F2_varDeg3_log2_lowerBound_withMIS = None                    
        with open(F2_estimates_withMIS_directory + problem + '.txt', 'r') as f_approxF2:
            for line in f_approxF2:
                if line.strip().split(" ")[0] == 'biregular_variable_degree_3_Tsol_1' and line.strip().split(" ")[2] == 'False': #not a timeout
                    F2_varDeg3_log2_lowerBound_withMIS = float(line.strip().split(" ")[4])
                    F2_varDeg3_time_withMIS = float(line.strip().split(" ")[6])

        #get computation time of MIS
        if F2_varDeg3_log2_lowerBound_withMIS is not None:
            if(os.path.isfile(sat_problems_withMIS_directory + problem + '.cnf.gz.no_w.cnf')):
                with open(sat_problems_withMIS_directory + problem + '.cnf.gz.no_w.cnf', 'r') as f:             
                    for line in f:
                        assert(line.startswith('c found the independent set in')) #should be the first line
                        independent_set_computation_time = float(line.split()[6])
                        F2_varDeg3_time_withMIS += independent_set_computation_time
                        break

                if F2_varDeg3_log2_lowerBound is None:
                    F2_varDeg3_time = F2_varDeg3_time_withMIS
                    F2_varDeg3_log2_lowerBound = F2_varDeg3_log2_lowerBound_withMIS
                elif F2_varDeg3_time_withMIS < F2_varDeg3_time:
                    F2_varDeg3_time = F2_varDeg3_time_withMIS
                    # F2_varDeg3_log2_lowerBound = F2_varDeg3_log2_lowerBound_withMIS
                    assert(np.abs(F2_varDeg3_log2_lowerBound - F2_varDeg3_log2_lowerBound_withMIS) < 64), ("F2 with and without independent sets differs by more than expected:", F2_varDeg3_log2_lowerBound, F2_varDeg3_log2_lowerBound_withMIS)
#                     print("F2 with independed set is faster than without")
                else:
                    pass
#                     print("F2 with independed set is SLOWER than without")
            else:
                print("MIS sat problem missing for:", problem)

    if F2_varDeg3_log2_lowerBound is not None:
        F2_lnZ_est = F2_varDeg3_log2_lowerBound/math.log(math.e, 2)
    else:
        F2_lnZ_est = None
        F2_time = None
        
    return F2_lnZ_est, F2_varDeg3_time

def get_exact_ln_modelCount(problem, SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/exact_SAT_counts_noIndSets/"):
    count_file = problem_name + '.txt'        
    ###### get the exact log solution count ######
    exact_ln_solution_count = None
    with open(SOLUTION_COUNTS_DIR + "/" + count_file, 'r') as f_solution_count:
        for line in f_solution_count:
            if line.strip().split(" ")[0] == 'dsharp':
                dsharp_solution_count = Decimal(line.strip().split(" ")[4])
                dsharp_time = float(line.strip().split(" ")[6])
                if not Decimal.is_nan(dsharp_solution_count):
                    exact_ln_solution_count = float(dsharp_solution_count.ln())
    assert(exact_ln_solution_count is not None) #we should have the exact count for all problems in the training/test set
    return exact_ln_solution_count, dsharp_time
                        


def get_baseline_info(problem_category, SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/exact_SAT_counts_noIndSets/",
                     LBP_EST_DIR = "/atlas/u/jkuck/learn_BP/data/LBP_estimates/"):
    '''
    
    Outputs:
    - dsharp_solving_times (list of floats): dsharp runtimes for all problems in problem_category
    - approxMC_se_list (list of floats): squared error between lnZ estimate by approxMC and exact lnZ
        for all problems in problem_category that approxMC finished within the timeout (5k seconds)
    - approxMC_runtime_list (list of floats): runtime for approxMC (best of w/out MIS and (w/ MIS + MIS computation time))
        for all problems in problem_category.  We use runtime of 5k seconds for problems that timed out.
    - F2_se_list (list of floats): squared error between lnZ estimate by F2 and exact lnZ
        for all problems in problem_category that F2 finished within the timeout (5k seconds)
    - F2_runtime_list (list of floats): runtime for F2 (best of w/out MIS and (w/ MIS + MIS computation time))
        for all problems in problem_category.  We use runtime of 5k seconds for problems that timed out.
    - LBP_se_list (list of floats): squared error between libdai loopy belief propagation estimate and exact lnZ
        for all problems in problem_category that loopy belief propagation did not crash on.
    '''
    problem_names = [problem['problem'] for problem in ALL_TRAIN_PROBLEMS[problem_category]]
    dsharp_solving_times = [problem['dsharp_time'] for problem in ALL_TRAIN_PROBLEMS[problem_category]]
    
    LBP_se_list = []
    
    approxMC_se_list = []
    approxMC_runtime_list = []
    
    F2_se_list = []
    F2_runtime_list = []   
    
    for problem_name in problem_names:
        count_file = problem_name + '.txt'        
        ###### get the exact log solution count ######
        exact_ln_solution_count = None
        with open(SOLUTION_COUNTS_DIR + "/" + count_file, 'r') as f_solution_count:
            for line in f_solution_count:
                if line.strip().split(" ")[0] == 'dsharp':
                    dsharp_solution_count = Decimal(line.strip().split(" ")[4])
                    if not Decimal.is_nan(dsharp_solution_count):
                        exact_ln_solution_count = float(dsharp_solution_count.ln())
        assert(exact_ln_solution_count is not None) #we should have the exact count for all problems in the training/test set
                        
        ###### get the LBP estimate ######
        LBP_estimate = None
        if os.path.isfile(LBP_EST_DIR + "/" + count_file):
            with open(LBP_EST_DIR + "/" + count_file, 'r') as f_lbp_est:
                for line in f_lbp_est:
                    if line.startswith('LBP_est_exact_ln_solution_count:'):
                        LBP_estimate = float(line.split()[1])
      
        if LBP_estimate is not None:
            LBP_se_list.append((LBP_estimate - exact_ln_solution_count)**2)
        
        ###### get the approxMC estimate ######
        approxMC_lnZ_est, approxMC_time = get_approxMC_estAndTime(problem=problem_name)
        if approxMC_lnZ_est is not None:
            approxMC_se_list.append((approxMC_lnZ_est - exact_ln_solution_count)**2)
            approxMC_runtime_list.append(approxMC_time)
        else:
            approxMC_runtime_list.append(5000)
            
        ###### get the F2 estimate ######
        F2_lnZ_est, F2_varDeg3_time = get_F2_estAndTime(problem=problem_name)
        if F2_lnZ_est is not None:
            F2_se_list.append((F2_lnZ_est - exact_ln_solution_count)**2)
            F2_runtime_list.append(F2_varDeg3_time)
        else:
            F2_runtime_list.append(5000)            
        
    return dsharp_solving_times, approxMC_se_list, approxMC_runtime_list, F2_se_list, F2_runtime_list, LBP_se_list

def make_baseline_table():
#     print("Benchma")
    # Benchmark Category & approxMC RMSE (completion %) & F2 RMSE (completion %) & min/70percentile/max dsharp time & min/mean/max F2 time & min/mean/max approxMC time
    for problem_category in PROBLEM_CATEGORY_TRAIN_options:
        dsharp_solving_times, approxMC_se_list, approxMC_runtime_list, F2_se_list, F2_runtime_list, LBP_se_list = get_baseline_info(problem_category)
        problem_category = ['\_' if l == '_' else l for l in problem_category]
        problem_category = ''.join(problem_category)
        print("%s & "  % (problem_category), end='')
        approxMC_completionFraction = len(approxMC_se_list)/len(approxMC_runtime_list)
        F2_completionFraction = len(F2_se_list)/len(F2_runtime_list)

        print("%.2f (%d\\%%) & " % (np.sqrt(np.mean(approxMC_se_list)), 100*approxMC_completionFraction), end='')
        print("%.1f (%d\\%%) & " % (np.sqrt(np.mean(F2_se_list)), 100*F2_completionFraction), end='')

        print("%.1f / %.1f / %.1f / %.1f & " % (np.min(dsharp_solving_times), np.percentile(dsharp_solving_times, 70), np.percentile(dsharp_solving_times, 85), np.max(dsharp_solving_times)), end='')
        print("%.1f / %.1f / %.1f & " % (np.min(approxMC_runtime_list), np.percentile(approxMC_runtime_list, 70), np.max(approxMC_runtime_list)), end='')
        print("%.1f / %.1f / %.1f & " % (np.min(F2_runtime_list), np.percentile(F2_runtime_list, 70), np.max(F2_runtime_list)), end='')
        print("\\\\")
        
        
def make_BPNN_table(dict_of_results):
    print(dict_of_results)
    for problem_category in PROBLEM_CATEGORY_TRAIN_options:
        if isinstance(problem_category, list):
            problem_category = problem_category[0]
        dsharp_solving_times, approxMC_se_list, approxMC_runtime_list, F2_se_list, F2_runtime_list, LBP_se_list = get_baseline_info(problem_category)
        for idx, train_val_split in enumerate(train_val_split_options):
            if idx == 0:
                print("\\multirow{ 2}{*}{%s} & %s &" % (problem_category, train_val_split), end = '')               
            else:
                print("& %s &" % (train_val_split), end = '')   
            for MSG_PASSING_ITERS in MSG_PASSING_ITERS_options:
                if ((problem_category, train_val_split, MSG_PASSING_ITERS) in dict_of_results):
                    print("%.2f / %.2f &" % (dict_of_results[(problem_category, train_val_split, MSG_PASSING_ITERS)]["RMSE_training"],\
                                      dict_of_results[(problem_category, train_val_split, MSG_PASSING_ITERS)]["RMSE_val"]), end = '')                
                else:
                    print("- / - &" % (), end = '')       
            
            print("\\\\")
        print("\\midrule")
    
                    
if __name__ == "__main__":
    make_baseline_table()

#     dict_of_results = get_dict_of_results(wandb_results_csv_finished)
#     make_BPNN_table(dict_of_results)