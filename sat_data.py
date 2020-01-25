import os
import math
from torch.utils.data import Dataset
from collections import defaultdict
from factor_graph import build_factorgraph_from_SATproblem
from utils import dotdict
import numpy as np

def parse_dimacs(filename, verbose=False):
    clauses = []
    dictionary_of_vars = defaultdict(int)
    # print("parse_dimacs, filename:", filename)  
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split()
            if len(line_as_list) == 0:
                continue
            # print("line_as_list[:-1]:")
            # print(line_as_list[:-1])
            # print()            
            if line_as_list[0] == "p":
                n_vars = int(line_as_list[2])
                n_clauses = int(line_as_list[3])
            elif line_as_list[0] == "c":
                continue
            else:
                cur_clause = [int(s) for s in line_as_list[:-1]]
                for var in cur_clause:
                    dictionary_of_vars[int(abs(var))] += 1
                clauses.append(cur_clause)
    # assert(n_clauses == len(clauses)), (n_clauses, len(clauses), filename)
    if(n_clauses != len(clauses)):
        if verbose:
            print("actual clause count doesn't match expected clause count!!")
    
    #make sure that all variables are named something in [1,...,n_vars]
    for var_name, var_degree in dictionary_of_vars.items():
        assert(var_name <= n_vars and var_name >= 1)

    #create dummy clauses for variables that don't explcitly appear, i.e. if 
    #variable 8 never appears explicitly in a clause this is equivalent to having
    #the additional clause (-8 8 0)
    for var_name in range(1, n_vars+1):
        if var_name not in dictionary_of_vars:
            clauses.append([-var_name, var_name])
            dictionary_of_vars[var_name] = 2
            # print("appended clause:", [-var_name, var_name])
            
# ######DEBUGGING#########
#     N = -1 #number of variables
#     max_var_degree = -1
#     for var_name, var_degree in dictionary_of_vars.items():
#         if var_name > N:
#             N = var_name
#         if var_degree > max_var_degree:
#             max_var_degree = var_degree
#     for var_name, var_degree in dictionary_of_vars.items():
#         if var_name > N:
#             N = var_name
#         if var_degree > max_var_degree:
#             max_var_degree = var_degree
#     if(N != len(dictionary_of_vars)):
#         for var_idx in range(1, N+1):
#             if var_idx not in dictionary_of_vars:
#                 print(var_idx, "missing from dictionary_of_vars")
#     assert(N == len(dictionary_of_vars)), (N, len(dictionary_of_vars), n_vars)



#     dictionary_of_vars_check = defaultdict(int)
#     for clause_idx, clause in enumerate(clauses):
#         for literal in clause:
#             dictionary_of_vars_check[np.abs(literal)] += 1

#     # print("a")

#     # check largest variable name equals the number of variables
#     N = -1 #number of variables
#     max_var_degree = -1
#     for var_name, var_degree in dictionary_of_vars_check.items():
#         if var_name > N:
#             N = var_name
#         if var_degree > max_var_degree:
#             max_var_degree = var_degree
#     if(N != len(dictionary_of_vars_check)):
#         for var_idx in range(1, N+1):
#             if var_idx not in dictionary_of_vars_check:
#                 print(var_idx, "missing from dictionary_of_vars_check")
#     assert(N == len(dictionary_of_vars_check)), (N, len(dictionary_of_vars_check))


# ######END DEBUGGING######

    # assert(len(dictionary_of_vars) == n_vars), (len(dictionary_of_vars), n_vars) #make sure we actually have this many variables

    # if (len(dictionary_of_vars) == n_vars):
    if True: #missing variables imply an always true clause, e.g. (-8 8 0) if 8 is missing
        if verbose:
            print("variable count checks succeeded")
        load_successful = True
    else:
        if verbose:
            print("variable count check failed")
        load_successful = False
        print("load failed for:", filename)
        print("len(dictionary_of_vars):", len(dictionary_of_vars))
        print("n_vars:", n_vars)
        for i in range(1, n_vars+1):
            if i not in dictionary_of_vars:
                print(i, "missing from dictionary_of_vars")
        print()
    return n_vars, clauses, load_successful


class SatProblems(Dataset):
    def __init__(self, counts_dir_name, problems_dir_name, dataset_size, begin_idx=0, verbose=True, epsilon=0, max_factor_dimensions=5):
        '''
        Inputs:
        - problems_dir_name (string): directory containing problems in cnf form 
        - counts_dir_name (string): directory containing .txt files with model counts for problems
            File name format: problem1.txt
            File content format: "sharpSAT time_out: False solution_count: 2097152 sharp_sat_time: 0.0"

        - begin_idx: (int) discard the first begin_idx problems, e.g. for validation
        - epsilon (float): set factor states with potential 0 to epsilon for numerical stability
        - max_factor_dimensions (int): do not construct a factor graph if the largest factor (clause) contains
            more than this many variables        
        '''
        # print("HI!!")
        self.sat_problems = []
        self.log_solution_counts = []
        problems_in_dir = os.listdir(problems_dir_name)
        # print("problems_in_dir:", problems_in_dir)
        # sleep(temp)
        discarded_count = 0

        load_failure_count = 0 # the number of SAT problems we failed to load properly
        no_solution_count = 0 # the number of SAT problems with no solutions
        unsolved_count = 0 # the number of SAT problems that we don't have an exact solution count for
        factors_to_large_count = 0 # the number of SAT problems with more than max_factor_dimensions variables in a clause
        dsharp_sharpsat_disagree = 0 # the number of SAT problems where dsharp and sharpsat disagree on the number of satisfying solutions

        for count_file in os.listdir(counts_dir_name):
            # if count_file[:5] == 'or-60':
            # if count_file[:2] == 'or':
                # continue
            if len(self.sat_problems) == dataset_size:
                break
            if count_file[-4:] != '.txt':
                print('not a text file')
                continue
            # problem_file = count_file[:-19] + '.cnf'
            problem_file = count_file[:-4] + '.cnf.gz.no_w.cnf'
            if problem_file not in problems_in_dir:
                if verbose:
                    print('no corresponding sat file for', problem_file, "count_file:", count_file)
                continue

            with open(counts_dir_name + "/" + count_file, 'r') as f_solution_count:
                sharpSAT_solution_count = None
                dsharp_solution_count = None
                for line in f_solution_count:
                    if line.strip().split(" ")[0] == 'sharpSAT':
                        sharpSAT_solution_count = float(line.strip().split(" ")[4])
                        if np.isnan(sharpSAT_solution_count):
                            sharpSAT_solution_count = None 
                    if line.strip().split(" ")[0] == 'dsharp':
                        dsharp_solution_count = float(line.strip().split(" ")[4])
                        if np.isnan(dsharp_solution_count):
                            dsharp_solution_count = None 


                if (dsharp_solution_count is not None) and (sharpSAT_solution_count is not None):
                    # assert(dsharp_solution_count == sharpSAT_solution_count), (dsharp_solution_count, sharpSAT_solution_count)
                    if dsharp_solution_count != sharpSAT_solution_count:
                        dsharp_sharpsat_disagree += 1
                        continue
                if dsharp_solution_count is not None:
                    solution_count = dsharp_solution_count
                elif sharpSAT_solution_count is not None:
                    solution_count = sharpSAT_solution_count
                else:
                    solution_count = None

            # assert(solution_count is not None)
            if solution_count is None:
                unsolved_count += 1
                continue

            if solution_count == 0:
                no_solution_count += 1
                continue
            log_solution_count = math.log(solution_count)    
            n_vars, clauses, load_successful = parse_dimacs(problems_dir_name + "/" + problem_file)
            if not load_successful:
                load_failure_count += 1
                continue
            # print("factor_graph:", factor_graph)
            if discarded_count == begin_idx:
                # print('using problem:', problem_file)
                factor_graph = build_factorgraph_from_SATproblem(clauses, epsilon=epsilon, max_factor_dimensions=max_factor_dimensions)
                if factor_graph is None: #largest clause contains too many variables
                    factors_to_large_count += 1
                    continue
                self.sat_problems.append(factor_graph)
                self.log_solution_counts.append(log_solution_count)
            else:
                discarded_count += 1
            assert(discarded_count <= begin_idx)
        assert(len(self.log_solution_counts) == len(self.sat_problems))
        print(len(self.log_solution_counts), "SAT problems loaded successfully")
        print(unsolved_count, "unsolved SAT problems")
        print(no_solution_count, "SAT problems with no solution (not loaded)")
        print(load_failure_count, "SAT problems failed to load properly")
        print(factors_to_large_count, "SAT problems have more than 5 variables in a clause")

    def __len__(self):
        return len(self.log_solution_counts)

    def __getitem__(self, index):
        '''
        Outputs:
        - sat_problem (FactorGraph, defined in models): factor graph representation of sat problem
        - log_solution_count (float): ln(# of satisfying solutions to the sat problem)
        '''
        sat_problem = self.sat_problems[index]
        log_solution_count = self.log_solution_counts[index]
        return sat_problem, log_solution_count





        