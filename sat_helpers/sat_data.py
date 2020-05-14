import os
import math
from torch.utils.data import Dataset
from collections import defaultdict
from factor_graph import FactorGraphData
from utils import dotdict
import numpy as np
from decimal import Decimal
import torch
import multiprocessing as mp
import psutil
from joblib import Parallel, delayed
from torch_geometric.data import InMemoryDataset

from parameters import LN_ZERO
from data.SAT_train_test_split import ALL_TRAIN_PROBLEMS, ALL_TEST_PROBLEMS


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

def get_SATproblems_list(problems_to_load, counts_dir_name, problems_dir_name, dataset_size=None, begin_idx=0, verbose=True, epsilon=0, 
                     max_factor_dimensions=5, return_logZ_list=False, belief_repeats=None):
    '''
    Inputs:
    - problems_to_load (list of strings): problems to load
    - problems_dir_name (string): directory containing problems in cnf form 
    - counts_dir_name (string): directory containing .txt files with model counts for problems
        File name format: problem1.txt
        File content format: "sharpSAT time_out: False solution_count: 2097152 sharp_sat_time: 0.0"

    - begin_idx: (int) discard the first begin_idx problems, e.g. for validation
    - epsilon (float): set factor states with potential 0 to epsilon for numerical stability
    - max_factor_dimensions (int): do not construct a factor graph if the largest factor (clause) contains
        more than this many variables      
    - dataset_size (int): return a maximum dataset_size SAT problems.  If none, return all problems  

    Outputs:
    - sat_problems (list of FactorGraphData): list of sat problems represented as factor graphs
    '''
    sat_problems = []
    ln_solution_counts = []

    discarded_count = 0

    load_failure_count = 0 # the number of SAT problems we failed to load properly
    no_solution_count = 0 # the number of SAT problems with no solutions
    unsolved_count = 0 # the number of SAT problems that we don't have an exact solution count for
    factors_to_large_count = 0 # the number of SAT problems with more than max_factor_dimensions variables in a clause
    dsharp_sharpsat_disagree = 0 # the number of SAT problems where dsharp and sharpsat disagree on the number of satisfying solutions

    problems_in_cnf_dir = os.listdir(problems_dir_name)
    # print("problems_in_cnf_dir:", problems_in_cnf_dir)
    problems_in_counts_dir = os.listdir(counts_dir_name)


    for problem_name in problems_to_load:
        if (dataset_size is not None) and (len(sat_problems) == dataset_size):
            break
        # problem_file = problem_name[:-19] + '.cnf'
        problem_file = problem_name + '.cnf.gz.no_w.cnf'
        if problem_file not in problems_in_cnf_dir:
            if verbose:
                print('no corresponding cnf file for', problem_file, "problem_name:", problem_name)
            continue
        count_file = problem_name + '.txt'
        if count_file not in problems_in_counts_dir:
            if verbose:
                print('no corresponding sat count file for', count_file, "problem_name:", problem_name)
            continue            

        with open(counts_dir_name + "/" + count_file, 'r') as f_solution_count:
            sharpSAT_solution_count = None
            dsharp_solution_count = None
            for line in f_solution_count:
                #Only use dsharp counts because dsharp and sharpSAT seem to disagree on some benchmarks
                #dsharp is consistent with randomized hashing methods while sharpSAT is not
                #this is with sampling sets removed, not sure what is going on with sharpSAT
#                 if line.strip().split(" ")[0] == 'sharpSAT':
#                     sharpSAT_solution_count = Decimal(line.strip().split(" ")[4])
#                     if Decimal.is_nan(sharpSAT_solution_count):
#                         sharpSAT_solution_count = None
                if line.strip().split(" ")[0] == 'dsharp':
                    dsharp_solution_count = Decimal(line.strip().split(" ")[4])
                    if Decimal.is_nan(dsharp_solution_count):
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
        ln_solution_count = float(solution_count.ln())
        n_vars, clauses, load_successful = parse_dimacs(problems_dir_name + "/" + problem_file)
        if not load_successful:
            load_failure_count += 1
            continue
        # print("factor_graph:", factor_graph)
        if discarded_count == begin_idx:
            # print('using problem:', problem_file)
            factor_graph = build_factorgraph_from_SATproblem(clauses, epsilon=epsilon, max_factor_dimensions=max_factor_dimensions, ln_Z=ln_solution_count, belief_repeats=belief_repeats)
            if factor_graph is None: #largest clause contains too many variables
                factors_to_large_count += 1
                continue
            sat_problems.append(factor_graph)
            ln_solution_counts.append(ln_solution_count)
            print("successfully loaded:", problem_name)
        else:
            discarded_count += 1
        assert(discarded_count <= begin_idx)
    assert(len(ln_solution_counts) == len(sat_problems))
    print(len(ln_solution_counts), "SAT problems loaded successfully")
    print(unsolved_count, "unsolved SAT problems")
    print(no_solution_count, "SAT problems with no solution (not loaded)")
    print(load_failure_count, "SAT problems failed to load properly")
    print(factors_to_large_count, "SAT problems have more than 5 variables in a clause")    
    if return_logZ_list:
        return sat_problems, ln_solution_counts
    else:
        return sat_problems    
    

class SATDataset(InMemoryDataset):    
    def __init__(self, root, dataset_type, problem_category, belief_repeats, epsilon=0, max_factor_dimensions=5,\
                 transform=None, pre_transform=None, SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/exact_SAT_counts_noIndSets/",\
                 SAT_PROBLEMS_DIR = "/atlas/u/jkuck/learn_BP/data/sat_problems_noIndSets"):
        '''
        Inputs:
        - dataset_type (string): 'train', 'val', or 'test'
        - problem_category (string): 'or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', or 'problems_90'
        - epsilon (float): set factor states with potential 0 to epsilon for numerical stability
        - max_factor_dimensions (int): do not construct a factor graph if the largest factor (clause) contains
            more than this many variables        
        '''      
        self.dataset_type = dataset_type
        self.problem_category = problem_category
        self.belief_repeats = belief_repeats
        self.epsilon = epsilon
        self.max_factor_dimensions = max_factor_dimensions
        self.SOLUTION_COUNTS_DIR = SOLUTION_COUNTS_DIR
        self.SAT_PROBLEMS_DIR = SAT_PROBLEMS_DIR

        assert(self.dataset_type in ['train', 'test'])
        if self.dataset_type == 'test':
            self.problem_names = [benchmark['problem'] for benchmark in ALL_TEST_PROBLEMS[problem_category]]
        else:
            self.problem_names = [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[problem_category]]


        super(SATDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['unused']
        pass
        # print("HI, looking for raw files :)")
        # # return ['some_file_1', 'some_file_2', ...]
        # dataset_file = './' + self.root + '/' + self.dataset_type + '%d_%d_%d_%.2f_%.2f_attField=%s.pkl' % (self.datasize, self.N_MIN, self.N_MAX, self.F_MAX, self.C_MAX, self.ATTRACTIVE_FIELD)
        # print("dataset_file:", dataset_file)
        # return [dataset_file]

    @property
    def processed_file_names(self):
        processed_dataset_file = self.dataset_type + '_%s_%d_%f_%d_pyTorchGeomProccesed.pt' %\
                                 (self.problem_category, self.belief_repeats, self.epsilon, self.max_factor_dimensions)
        
        return [processed_dataset_file]

    def download(self):
        pass
        # assert(False), "Error, need to generate new data!!"
        # Download to `self.raw_dir`.

    def process(self):
        # get a list of sat problems as factor graphs (FactorGraphData objects)    
        SAT_problem_list = get_SATproblems_list(problems_to_load=self.problem_names, counts_dir_name=self.SOLUTION_COUNTS_DIR,\
                           problems_dir_name=self.SAT_PROBLEMS_DIR, verbose=True, epsilon=self.epsilon, max_factor_dimensions=self.max_factor_dimensions,\
                           return_logZ_list=False, belief_repeats=self.belief_repeats)


        data, slices = self.collate(SAT_problem_list)
        torch.save((data, slices), self.processed_paths[0])




#loading in parallel seems to actually hurt performance.  Instead use the pytorch geometric dataset to preprocess/load fast
# def get_SATproblems_list_parallel_helper(problem_name, problems_in_cnf_dir, problems_in_counts_dir, counts_dir_name, problems_dir_name,\
def get_SATproblems_list_parallel_helper(problem_name, counts_dir_name, problems_dir_name,\
                                         epsilon, max_factor_dimensions, belief_repeats, verbose):
    return_dictionary = {"sat_problem": None,
                         "ln_solution_count": None,
                         "load_failure_count": False,
                         "no_solution_count": False,
                         "unsolved_count": False,
                         "factors_to_large_count": False,
                         "dsharp_sharpsat_disagree": False,
                         }

    problem_file = problem_name + '.cnf.gz.no_w.cnf'
    # if problem_file not in problems_in_cnf_dir:
    #     if verbose:
    #         print('no corresponding cnf file for', problem_file, "problem_name:", problem_name)
    #     return return_dictionary
    count_file = problem_name + '.txt'
    # if count_file not in problems_in_counts_dir:
    #     if verbose:
    #         print('no corresponding sat count file for', count_file, "problem_name:", problem_name)
    #     return return_dictionary            

    with open(counts_dir_name + "/" + count_file, 'r') as f_solution_count:
        sharpSAT_solution_count = None
        dsharp_solution_count = None
        for line in f_solution_count:
            #Only use dsharp counts because dsharp and sharpSAT seem to disagree on some benchmarks
            #dsharp is consistent with randomized hashing methods while sharpSAT is not
            #this is with sampling sets removed, not sure what is going on with sharpSAT
#                 if line.strip().split(" ")[0] == 'sharpSAT':
#                     sharpSAT_solution_count = Decimal(line.strip().split(" ")[4])
#                     if Decimal.is_nan(sharpSAT_solution_count):
#                         sharpSAT_solution_count = None
            if line.strip().split(" ")[0] == 'dsharp':
                dsharp_solution_count = Decimal(line.strip().split(" ")[4])
                if Decimal.is_nan(dsharp_solution_count):
                    dsharp_solution_count = None 


        if (dsharp_solution_count is not None) and (sharpSAT_solution_count is not None):
            # assert(dsharp_solution_count == sharpSAT_solution_count), (dsharp_solution_count, sharpSAT_solution_count)
            if dsharp_solution_count != sharpSAT_solution_count:
                return_dictionary["dsharp_sharpsat_disagree"] = True
                return return_dictionary
        if dsharp_solution_count is not None:
            solution_count = dsharp_solution_count
        elif sharpSAT_solution_count is not None:
            solution_count = sharpSAT_solution_count
        else:
            solution_count = None

    # assert(solution_count is not None)
    if solution_count is None:
        return_dictionary["unsolved_count"] = True
        return return_dictionary

    if solution_count == 0:
        return_dictionary["no_solution_count"] = True
        return return_dictionary

    ln_solution_count = float(solution_count.ln())
    n_vars, clauses, load_successful = parse_dimacs(problems_dir_name + "/" + problem_file)
    if not load_successful:
        return_dictionary["load_failure_count"] = True
        return return_dictionary
        
    # print("factor_graph:", factor_graph)
    # print('using problem:', problem_file)
    factor_graph = build_factorgraph_from_SATproblem(clauses, epsilon=epsilon, max_factor_dimensions=max_factor_dimensions, ln_Z=ln_solution_count, belief_repeats=belief_repeats)
    if factor_graph is None: #largest clause contains too many variables
        return_dictionary["factors_to_large_count"] = True
        return return_dictionary            

    
    print("successfully loaded:", problem_name)    
    return_dictionary["sat_problem"] = factor_graph
    return_dictionary["ln_solution_count"] = ln_solution_count
    return return_dictionary      

def get_SATproblems_list_parallel(problems_to_load, counts_dir_name, problems_dir_name, dataset_size, begin_idx=0, verbose=True, epsilon=0, 
                     max_factor_dimensions=5, return_logZ_list=False, belief_repeats=None):
    '''
    parallel version of get_SATproblems_list
    Inputs:
    - problems_to_load (list of strings): problems to load
    - problems_dir_name (string): directory containing problems in cnf form 
    - counts_dir_name (string): directory containing .txt files with model counts for problems
        File name format: problem1.txt
        File content format: "sharpSAT time_out: False solution_count: 2097152 sharp_sat_time: 0.0"

    - dataset_size (int): number of problems to return

    - begin_idx: (int) discard the first begin_idx problems, e.g. for validation
    - epsilon (float): set factor states with potential 0 to epsilon for numerical stability
    - max_factor_dimensions (int): do not construct a factor graph if the largest factor (clause) contains
        more than this many variables        
    '''
    sat_problems = []
    ln_solution_counts = []

    problems_in_cnf_dir = os.listdir(problems_dir_name)
    # print("problems_in_cnf_dir:", problems_in_cnf_dir)
    problems_in_counts_dir = os.listdir(counts_dir_name)

    num_cpu = psutil.cpu_count(logical = False)
    num_cores = 4

    print("loading sat problems using", num_cores, "cores")
    for idx in range(begin_idx, len(problems_to_load), num_cores):
        processed_list = Parallel(n_jobs=num_cores, prefer="threads")(\
            # delayed(get_SATproblems_list_parallel_helper)(problem_name=problem_name, problems_in_cnf_dir=problems_in_cnf_dir,\
            #                                              problems_in_counts_dir=problems_in_counts_dir, counts_dir_name=counts_dir_name,\
            delayed(get_SATproblems_list_parallel_helper)(problem_name=problem_name, counts_dir_name=counts_dir_name,\
                                                         problems_dir_name=problems_dir_name, epsilon=epsilon,\
                                                         max_factor_dimensions=max_factor_dimensions, belief_repeats=belief_repeats, verbose=verbose) 
            for problem_name in problems_to_load[idx:idx+num_cores])
        cur_sat_problems = [d['sat_problem'] for d in processed_list if (d['sat_problem'] is not None)]
        cur_ln_solution_counts = [d['ln_solution_count'] for d in processed_list if (d['ln_solution_count'] is not None)]
        assert(len(cur_sat_problems) == len(cur_ln_solution_counts)), (len(cur_sat_problems), len(cur_ln_solution_counts))
        sat_problems.extend(cur_sat_problems)
        ln_solution_counts.extend(cur_ln_solution_counts)
        if len(sat_problems) >= dataset_size:
            break
        print("finished iteration :)")

        
    assert(len(ln_solution_counts) == len(sat_problems))
    print(len(ln_solution_counts), "SAT problems loaded successfully")    
    if return_logZ_list:
        return sat_problems[:dataset_size], ln_solution_counts[:dataset_size]
    else:
        return sat_problems[:dataset_size]    
        
    
#Pytorch dataset
class SatProblems(Dataset):
    def __init__(self, problems_to_load, counts_dir_name, problems_dir_name, dataset_size, begin_idx=0, verbose=True, epsilon=0, max_factor_dimensions=5):
        '''
        Inputs:
        - problems_to_load (list of strings): problems to load
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
        # self.sat_problems, self.ln_solution_counts = get_SATproblems_list(problems_to_load=problems_to_load, counts_dir_name=counts_dir_name,\
        #     problems_dir_name=problems_dir_name, dataset_size=dataset_size, begin_idx=begin_idx, verbose=verbose, epsilon=epsilon, 
        #              max_factor_dimensions=max_factor_dimensions, return_logZ_list=True)

        self.sat_problems, self.ln_solution_counts = get_SATproblems_list_parallel(problems_to_load=problems_to_load, counts_dir_name=counts_dir_name,\
            problems_dir_name=problems_dir_name, dataset_size=dataset_size, begin_idx=begin_idx, verbose=verbose, epsilon=epsilon, 
                     max_factor_dimensions=max_factor_dimensions, return_logZ_list=True)
                     
    def __len__(self):
        return len(self.ln_solution_counts)

    def __getitem__(self, index):
        '''
        Outputs:
        - sat_problem (FactorGraphData, defined in factor_graph.py): factor graph representation of sat problem
        - ln_solution_count (float): ln(# of satisfying solutions to the sat problem)
        '''
        sat_problem = self.sat_problems[index]
        ln_solution_count = self.ln_solution_counts[index]
        return sat_problem, ln_solution_count

def build_factorPotential_fromClause(clause, state_dimensions, epsilon):
    '''
    The ith variable in a clause corresponds to the ith dimension in the tensor representation of the state.
    Inputs:
    - epsilon (float): set states with potential 0 to epsilon for numerical stability

    Outputs:
    - state (tensor): 1 for variable assignments that satisfy the clause, 0 otherwise
    - mask (tensor): 1 signifies an invalid location (outside factor's valid variables), 0 signifies a valid location 
    '''
    #Create a tensor for the 2^state_dimensions states
    state = torch.zeros([2 for i in range(state_dimensions)])
    mask = torch.zeros([2 for i in range(state_dimensions)])
    #Iterate over the 2^state_dimensions variable assignments and set those to 1 that satisfy the clause
    for indices in np.ndindex(state.shape):
        junk_location = False
        for dimension in range(len(clause), state_dimensions):
            if indices[dimension] == 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            mask[indices] = 1
            continue
        set_to_1 = False
        for dimension, index_val in enumerate(indices):
            if dimension >= len(clause):
                break
            if clause[dimension] > 0 and index_val == 1:
                set_to_1 = True
            elif clause[dimension] < 0 and index_val == 0:
                set_to_1 = True
        if set_to_1:
            state[indices] = 1
        else:
            state[indices] = epsilon
    return state, mask

def test_build_factorPotential_fromClause():
    for new_dimensions in range(3,7):
        state, mask = build_factorPotential_fromClause([1, -2, 3], new_dimensions)
        print(state)
        for indices in np.ndindex(state.shape):
            print(tuple(reversed(indices)), state[tuple(reversed(indices))])


def build_edge_var_indices(clauses, max_clause_degree=None):
    print("max_clause_degree:", max_clause_degree)
    indices_at_source_node = []
    indices_at_destination_node = []
    for clause in clauses:
        for var_idx in range(len(clause)):
            indices_at_source_node.append(var_idx) #source node is the factor
            indices_at_destination_node.append(0) #destination node is the variable
            if max_clause_degree is not None:
                assert(var_idx < max_clause_degree), (var_idx, max_clause_degree)
    edge_var_indices = torch.tensor([indices_at_source_node, indices_at_destination_node])
    return edge_var_indices

def test_build_edge_var_indices():
    clauses = [[1, -2, 3], [-2, 4], [3]]
    edge_var_indices = build_edge_var_indices(clauses)
    expected_edge_var_indices = torch.tensor([[0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0]])
    assert(torch.all(torch.eq(edge_var_indices, expected_edge_var_indices)))
    print(edge_var_indices)
    print(expected_edge_var_indices)

def build_factorgraph_from_SATproblem(clauses, initialize_randomly=False, epsilon=0, max_factor_dimensions=5,
                                      local_state_dim=False, ln_Z=None, belief_repeats=None):
    '''
    Take a SAT problem in CNF form (specified by clauses) and return a factor graph representation
    whose partition function is the number of satisfying solutions

    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    - initialize_randomly: (bool) if true randomly initialize beliefs and previous messages
        if false initialize to 1
    - epsilon (float): set states with potential 0 to epsilon for numerical stability
    - max_factor_dimensions (int): do not construct a factor graph if the largest factor (clause) contains
        more than this many variables
    - local_state_dim (bool): if True, then the number of dimensions in each factor is set to the number of 
        variables in the largest clause in /this/ problem.  If False, then the number of dimensions in each factor
        is set to max_factor_dimensions for compatibility with other SAT problems.
    - ln_Z : natural logarithm of the partition function


    Outputs:
    - FactorGraphData (FactorGraphData): or None if there is a clause containing more than max_factor_dimensions variables
    '''
    num_factors = len(clauses)
    #list of [factor_idx, var_idx] for each edge factor to variable edge
    factorToVar_edge_index_list = []
    dictionary_of_vars = defaultdict(int)
    
    # factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
    # factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
    factorToVar_double_list = []
    
    for clause_idx, clause in enumerate(clauses):
        cur_clause_variable_indices = []
        for literal in clause:
            var_node_idx = np.abs(literal) - 1
            factorToVar_edge_index_list.append([clause_idx, var_node_idx])
            dictionary_of_vars[np.abs(literal)] += 1
            cur_clause_variable_indices.append(var_node_idx)
        factorToVar_double_list.append(cur_clause_variable_indices)

    # print("a")

    # check largest variable name equals the number of variables
    N = -1 #number of variables
    max_var_degree = -1
    for var_name, var_degree in dictionary_of_vars.items():
        if var_name > N:
            N = var_name
        if var_degree > max_var_degree:
            max_var_degree = var_degree
    if(N != len(dictionary_of_vars)):
        for var_idx in range(1, N+1):
            if var_idx not in dictionary_of_vars:
                print(var_idx, "missing from dictionary_of_vars")
    assert(N == len(dictionary_of_vars)), (N, len(dictionary_of_vars))
    print("max_var_degree:", max_var_degree)
    # print("b")

    # get largest clause degree
    max_clause_degree = -1
    for clause in clauses:
        if len(clause) > max_clause_degree:
            max_clause_degree = len(clause)
    if max_clause_degree > max_factor_dimensions:
        print("max_clause_degree too large:", max_clause_degree)
        return None
    # state_dimensions = max(max_clause_degree, max_var_degree)
    if local_state_dim:
        state_dimensions = max_clause_degree
    else:
        state_dimensions = max_factor_dimensions

    factorToVar_edge_index = torch.tensor(factorToVar_edge_index_list, dtype=torch.long)

    # print("c")


####################    # create local indices of variable nodes for each clause node
####################    # the first variable appearing in a clause has index 0, the second variable
####################    # appearing in a clause has index 1, etc.  It is important to keep track
####################    # of this because these variable indices are different between clauses
####################
####################    clause_node_variable_indices = []

    # factor_potentials = torch.stack([build_factorPotential_fromClause(clause=clause, state_dimensions=state_dimensions, epsilon=epsilon) for clause in clauses], dim=0)
    states = []
    masks = []
    for clause in clauses:
        state, mask = build_factorPotential_fromClause(clause=clause, state_dimensions=state_dimensions, epsilon=epsilon)
        states.append(state)
        masks.append(mask)
    factor_potentials = torch.stack(states, dim=0)
    factor_potential_masks = torch.stack(masks, dim=0)
 


   
    # print("d")


    x_base = torch.zeros_like(factor_potentials)
    x_base.copy_(factor_potentials)

    edge_var_indices = build_edge_var_indices(clauses, max_clause_degree=max_clause_degree)
    # print("state_dimensions:", state_dimensions)

    edge_count = edge_var_indices.shape[1]


    log_potentials = torch.log(factor_potentials)
    log_potentials[torch.where(log_potentials == -np.inf)] = LN_ZERO
    factor_graph = FactorGraphData(factor_potentials=log_potentials,
                 factorToVar_edge_index=factorToVar_edge_index.t().contiguous(), numVars=N, numFactors=num_factors, 
                 edge_var_indices=edge_var_indices, state_dimensions=state_dimensions,
                 factor_potential_masks=factor_potential_masks, ln_Z=ln_Z, factorToVar_double_list=factorToVar_double_list,
                 var_cardinality=2, belief_repeats=belief_repeats)

    return factor_graph




        