import os
import math
from collections import defaultdict
from models import build_factorgraph_from_SATproblem

def parse_dimacs(filename):
    clauses = []
    dictionary_of_vars = defaultdict(int)    
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split(" ")
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
        print("actual clause count doesn't match expected clause count!!")
    #check variables are numbered 1 to n_vars with no gaps        
    num_vars_check = -1
    for var_name, var_degree in dictionary_of_vars.items():
        if var_name > num_vars_check:
            num_vars_check = var_name
    assert(num_vars_check == n_vars) #make sure largest variable is named n_vars
    # assert(len(dictionary_of_vars) == n_vars), (len(dictionary_of_vars), n_vars) #make sure we actually have this many variables
    if len(dictionary_of_vars) == n_vars:
        print("variable count check succeeded")
        load_successful = True
    else:
        print("variable count check failed")
        load_successful = False

    return n_vars, clauses, load_successful

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SatProblems():
    def __init__(self, counts_dir_name, problems_dir_name, dataset_size):
        '''
        Inputs:
        - problems_dir_name (string): directory containing problems in cnf form 
        - counts_dir_name (string): directory containing .txt files with model counts for problems
            File name format: problem1.txt
            File content format: "sharpSAT time_out: False solution_count: 2097152 sharp_sat_time: 0.0"


        '''
        print("HI!!")
        self.sat_problems = []
        self.log_solution_counts = []
        problems_in_dir = os.listdir(problems_dir_name)
        print("problems_in_dir:", problems_in_dir)
        # sleep(temp)
        for count_file in os.listdir(counts_dir_name):
            if len(self.sat_problems) == dataset_size:
                break
            if count_file[-4:] != '.txt':
                print('not a text file')
                continue
            problem_file = count_file[:-19] + '.cnf'
            if problem_file not in problems_in_dir:
                print('no corresponding sat file for', problem_file)
                continue

            with open(counts_dir_name + "/" + count_file, 'r') as f_solution_count:
                solution_count = None
                for line in f_solution_count:
                    if line.strip().split(" ")[0] == 'sharpSAT':
                        solution_count = float(line.strip().split(" ")[4])
                        if solution_count > 0:
                            log_solution_count = math.log(solution_count)
            assert(solution_count is not None)
            if solution_count == 0:
                continue
            n_vars, clauses, load_successful = parse_dimacs(problems_dir_name + "/" + problem_file)
            if not load_successful:
                continue
            factor_graph = build_factorgraph_from_SATproblem(clauses)
            print("factor_graph:", factor_graph)
            self.sat_problems.append(factor_graph)
            self.log_solution_counts.append(log_solution_count)
            assert(len(self.log_solution_counts) == len(self.sat_problems))
        print(len(self.log_solution_counts), "SAT problems loaded")

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





        