import random
import numpy as np
import torch
# from factor_graph import FactorGraphData
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
import json
from os import path

from libDAI_utils import run_loopyBP

'''
Info by category
CAN USE
--------------------------------------------------------------------------------
problem_cat: ObjDetect LBP error = 0.1938839747

problems with a maximum factor dimension of 3
18 with variable cardinality = 16 maximum factor dimensionality = 2
20 with variable cardinality = 11 maximum factor dimensionality = 2
18 with variable cardinality = 21 maximum factor dimensionality = 2

--------------------------------------------------------------------------------
problem_cat: Segment LBP error = 1.71570308
problems with a maximum factor dimension of 3
35 with variable cardinality = 2 maximum factor dimensionality = 2

--------------------------------------------------------------------------------
problem_cat: Grids LBP error = 58.34310638
problems with a maximum factor dimension of 3
6 with variable cardinality = 2 maximum factor dimensionality = 2

--------------------------------------------------------------------------------
problem_cat: DBN LBP error = 4.618604028
problems with a maximum factor dimension of 3
47 with variable cardinality = 2 maximum factor dimensionality = 2

--------------------------------------------------------------------------------
problem_cat: Promedas LBP error = 0.2545449077
problems with a maximum factor dimension of 3
46 with variable cardinality = 2 maximum factor dimensionality = 3

MAYBE CAN USE, SMALL DATASET
--------------------------------------------------------------------------------
problem_cat: Grids LBP error = 

problems with a maximum factor dimension of 3

################################################################################
################################################################################
CANT USE, different variable cardinalities

--------------------------------------------------------------------------------
problem_cat: pedigree  LBP error = 2.23205032
16 problems with variables of different cardinalities: 

--------------------------------------------------------------------------------
problem_cat: CSP LBP error = 18.41479318
5 problems with variables of different cardinalities: 

problems with a maximum factor dimension of 3
2 with variable cardinality = 3 maximum factor dimensionality = 2
3 with variable cardinality = 4 maximum factor dimensionality = 2

--------------------------------------------------------------------------------
problem_cat: Protein LBP error = 0.0252224026
54 problems with variables of different cardinalities: 

'''

max_factor_dimension_by_groups = {
    'group1': 3,
    'group2': 2,
    'group3': 2,
    'group4': 2,
    'group5': 14,
    'group7': 3,
    }
    

def build_padded_factor_potential(padded_size, factor_potential):
    '''
    Inputs:
    - padded_size (list of ints): the output size after padding
    - factor_potential (tensor): the input factor potential before padding

    Outputs:
    - padded_factor_potential (tensor): 1 for variable assignments that satisfy the clause, 0 otherwise
    - mask (tensor): 1 signifies an invalid location (outside factor's valid variables), 0 signifies a valid location 
    '''
    input_dimensions = len(factor_potential.shape) #dimensions in the input factor potential
    padded_dimensions = len(padded_size) #dimensions in the output factor potential after padding
    assert(padded_dimensions > input_dimensions)

    assert(tuple(factor_potential.shape) == tuple(padded_size[:input_dimensions]))
    padded_factor_potential = torch.zeros(padded_size)
    mask = torch.zeros(padded_size)

    for indices in np.ndindex(padded_factor_potential.shape):
        junk_location = False
        for dimension in range(input_dimensions, padded_dimensions):
            if indices[dimension] == 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            mask[indices] = 1
            continue
        else:
            padded_factor_potential[indices] = factor_potential[indices[:input_dimensions]]
    return padded_factor_potential, mask

def process_uai_model(file_name='Promedus_36.uai',\
                      problem_directory='./UAI_inference_data/data/PR_prob/',\
                      solution_directory='./UAI_inference_data/data/PR_sol/',\
                      belief_repeats=1, max_allowed_factor_dimension=None,\
                      dataset_type='train'):
    '''

    Outputs:
    - factor_graph (FactorGraphData): factor graph representation, or None if all variables do not have the same cardinality
    - max_allowed_factor_dimension (int): do not construct a factor graph if the largest factor has more than this many dimensions.
        set factor dimensionality to this value
    - dataset_type (string): train/test
    '''
    # format info here: http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html
    with open(problem_directory+'/train_test_split.json', 'r') as json_file:
        all_problems = json.load(json_file)
        if dataset_type == 'train':
            problems = all_problems['train_problems']
        else:
            assert(dataset_type == 'test')
            problems = all_problems['test_problems']
        partition_function = None
        for problem in problems:
            if problem["name"] == file_name:
                partition_function = problem['exact_log_Z']
                break
        
    assert(partition_function is not None), "Error reading ground truth partition function"


    with open(problem_directory + file_name, 'r') as f:  
        preamble = True
        factor_entries = None
        factor_entry_count = None

        factor_idx = 0
        factorToVar_edge_index = []
        edge_var_indices = []
        # factorToVar_double_list[i][j] is the index of the jth variable in the ith factor 
        factorToVar_double_list = []
        factor_potentials = []
        
        #the number of variables in the largest factor
        max_factor_dimension = 0
        all_factor_dimensions = [] #all_factor_dimensions[i] is the number of variables in the ith factor

        for line_idx, line in enumerate(f):
            if line_idx == 0:
                assert(line.split()[0] == "MARKOV"), line
                continue
            if line_idx == 1:
                variable_count = int(line)
                continue
            if line_idx == 2:
                variable_cardinalities = line.split()
                #var_cardinalities[i] is the cardinality of the ith variable
                variable_cardinalities = [int(var_card) for var_card in variable_cardinalities]
                assert(len(variable_cardinalities) == variable_count)
                continue
            if line_idx == 3:
                factor_count = int(line)
                continue

            if preamble: #still processing the preamble
                split_line = line.split()
                if len(split_line) == 0:
                    continue
                factor_dimension = int(split_line[0]) #the number of variables in the factor
                all_factor_dimensions.append(factor_dimension)
                if factor_dimension > max_factor_dimension:
                    max_factor_dimension = factor_dimension
                var_indices = [int(var_idx) for var_idx in split_line[1:]] #indices of all variables in this factor
                assert(len(var_indices) == factor_dimension)
                #enumerate the variale indices within the factor and the global variable indices
                for var_idx_in_factor, var_idx in enumerate(var_indices): 
                    factorToVar_edge_index.append([factor_idx, var_idx])
                    edge_var_indices.append([var_idx_in_factor, -99])
                factorToVar_double_list.append(var_indices)
                factor_idx += 1

                if len(all_factor_dimensions) == factor_count: #done with the preamble
                    preamble = False
                    assert(factor_idx == factor_count)
                    factor_idx = -1 #reset for processing potentials
                    continue

            else: #done with preamble, processing potentials
                split_line = line.split()
                if len(split_line) == 0:
                    continue
                elif factor_entry_count is None:
                    # assert(len(split_line) == 1)
                    factor_entry_count = int(split_line[0])
                    factor_entries = []
                    factor_idx += 1
                    if len(split_line) > 1:
                        factor_entries.extend([float(factor_entry) for factor_entry in split_line[1:]])
                        assert(len(factor_entries) <= factor_entry_count)
                        if len(factor_entries) == factor_entry_count: #store potential
                            factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in factorToVar_double_list[factor_idx]]
                            factor_potential = torch.tensor(factor_entries).reshape(factor_var_cardinalities)
                            factor_potentials.append(factor_potential)
                            factor_entries = None
                            factor_entry_count = None                    
                else:
                    factor_entries.extend([float(factor_entry) for factor_entry in split_line])
                    assert(len(factor_entries) <= factor_entry_count)
                    if len(factor_entries) == factor_entry_count: #store potential
                        factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in factorToVar_double_list[factor_idx]]
                        factor_potential = torch.tensor(factor_entries).reshape(factor_var_cardinalities)
                        factor_potentials.append(factor_potential)
                        factor_entries = None
                        factor_entry_count = None
    assert(factor_entries is None)
    assert(factor_entry_count is None)
    assert(factor_idx + 1 == factor_count)
    assert(len(factor_potentials) == factor_count)

    #check if all variables have the same cardinality
    all_variables_share_cardinality = True
    var_cardinality = variable_cardinalities[0]
    for var_card in variable_cardinalities:
        if var_card != var_cardinality:
            all_variables_share_cardinality = False
            return None #haven't implemented variables with different cardinality

    if max_factor_dimension > max_allowed_factor_dimension:
        return None #this factor graph has factor with more dimensions than is allowed
    else:
        max_factor_dimension = max_allowed_factor_dimension #construct the factor graph representing every factor with max_allowed_factor_dimension dimensions
    # print("max_factor_dimension:", max_factor_dimension, "var_cardinality:", var_cardinality)
    # return True

    padded_factor_potentials = [] #pad factor potentials so they all have the same size
    factor_potential_masks = [] #store masks: 1 signifies an invalid location (padding), 0 signifies a valid location
    for factor_idx, factor_potential in enumerate(factor_potentials):
        if all_factor_dimensions[factor_idx] == max_factor_dimension:
            padded_factor_potentials.append(factor_potential)
            factor_potential_masks.append(torch.zeros([var_cardinality for i in range(max_factor_dimension)]))
        else:
            padded_size = [var_cardinality for i in range(max_factor_dimension)]
            padded_factor_potential, mask = build_padded_factor_potential(padded_size=padded_size, factor_potential=factor_potential)
            padded_factor_potentials.append(padded_factor_potential)
            factor_potential_masks.append(mask)

    factor_potentials = torch.stack(padded_factor_potentials, dim=0)
    factor_potential_masks = torch.stack(factor_potential_masks, dim=0)
    factorToVar_edge_index = torch.tensor(factorToVar_edge_index).t().contiguous()
    edge_var_indices = torch.tensor(edge_var_indices).t().contiguous()
    ln_Z = np.log(10)*partition_function #UAI stores partition function in log base 10, convert to log base e
    factor_graph = FactorGraphData(factor_potentials=torch.log(factor_potentials),
                 factorToVar_edge_index=factorToVar_edge_index, numVars=variable_count, numFactors=factor_count, 
                 edge_var_indices=edge_var_indices, state_dimensions=max_factor_dimension,
                 factor_potential_masks=factor_potential_masks, ln_Z=ln_Z, factorToVar_double_list=factorToVar_double_list,
                 var_cardinality=var_cardinality, belief_repeats=belief_repeats)
    return factor_graph




class UAI_Dataset(InMemoryDataset):    
    def __init__(self, root, dataset_type, problem_category, belief_repeats, max_factor_dimension,\
                 transform=None, pre_transform=None):
        '''
        Inputs:
        - dataset_type (string): 'train', 'val', or 'test'
        - problem_category (string): 'or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', or 'problems_90'
        - epsilon (float): set factor states with potential 0 to epsilon for numerical stability
        - max_factor_dimension (int): do not construct a factor graph if the largest factor has more than this many dimensions
        '''      
        self.dataset_type = dataset_type
        self.problem_category = problem_category
        self.belief_repeats = belief_repeats
        self.max_factor_dimension = max_factor_dimension
        self.PROBLEMS_DIR = root + problem_category + "/"

        assert(self.dataset_type in ['train', 'test'])
        # if problem_category[:5] == 'group':
        #     self.max_factor_dimension = max_factor_dimension_by_groups[problem_category]
        #     if self.dataset_type == 'test':
        #         self.problem_names = test_groups[problem_category]
        #     else:
        #         self.problem_names = train_groups[problem_category]

        with open(self.PROBLEMS_DIR +'train_test_split.json', 'r') as json_file:
            all_problems = json.load(json_file)
            if dataset_type == 'train':
                problems = all_problems['train_problems']
            else:
                assert(dataset_type == 'test')
                problems = all_problems['test_problems']

            self.problem_names = [problem["name"] for problem in problems]


        super(UAI_Dataset, self).__init__(root+'pytorchGeom_proccesed/', transform, pre_transform)

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
        processed_dataset_file = self.dataset_type + '_%s_%d_%d_pyTorchGeomProccesed.pt' %\
                                 (self.problem_category, self.belief_repeats, self.max_factor_dimension)
        
        return [processed_dataset_file]

    def download(self):
        pass
        # assert(False), "Error, need to generate new data!!"
        # Download to `self.raw_dir`.

    def process(self):
        # get a list of sat problems as factor graphs (FactorGraphData objects)    
        problem_list = [process_uai_model(file_name=problem_name,\
                                          problem_directory=self.PROBLEMS_DIR,\
                                          belief_repeats=self.belief_repeats,\
                                          max_allowed_factor_dimension=self.max_factor_dimension,
                                          dataset_type=self.dataset_type) for problem_name in self.problem_names]
        #remove problems that failded to load
        problem_list_final = [problem for problem in problem_list if (problem is not None)]

        data, slices = self.collate(problem_list_final)
        torch.save((data, slices), self.processed_paths[0])


def inspect_uai_model(file_name='BN_0.uai',\
                      problem_directory='./data/BN/',\
                      belief_repeats=1, RUN_LBP=True):
    '''
    Inpspect the specified UAI model

    Outputs:
    - variables_equal_cardinality (bool): True if the variables all have the same cardinality
    - var_cardinality (int): if variables_equal_cardinality=True, the cardinality of all variables
    - max_factor_dimension
    '''
    with open(problem_directory + file_name, 'r') as f:  
        preamble = True
        factor_entries = None
        factor_entry_count = None

        factor_idx = 0
        factorToVar_edge_index = []
        edge_var_indices = []
        # factorToVar_double_list[i][j] is the index of the jth variable in the ith factor 
        factorToVar_double_list = []
        factor_potentials = []
        
        #the number of variables in the largest factor
        max_factor_dimension = 0
        all_factor_dimensions = [] #all_factor_dimensions[i] is the number of variables in the ith factor

        for line_idx, line in enumerate(f):
            if line_idx == 0:
                assert(line.split()[0] == "MARKOV"), line
                continue
            if line_idx == 1:
                variable_count = int(line)
                continue
            if line_idx == 2:
                variable_cardinalities = line.split()
                #var_cardinalities[i] is the cardinality of the ith variable
                variable_cardinalities = [int(var_card) for var_card in variable_cardinalities]
                assert(len(variable_cardinalities) == variable_count)
                continue
            if line_idx == 3:
                factor_count = int(line)
                continue

            if preamble: #still processing the preamble
                split_line = line.split()
                if len(split_line) == 0:
                    continue
                factor_dimension = int(split_line[0]) #the number of variables in the factor
                all_factor_dimensions.append(factor_dimension)
                if factor_dimension > max_factor_dimension:
                    max_factor_dimension = factor_dimension
                var_indices = [int(var_idx) for var_idx in split_line[1:]] #indices of all variables in this factor
                assert(len(var_indices) == factor_dimension)
                #enumerate the variale indices within the factor and the global variable indices
                for var_idx_in_factor, var_idx in enumerate(var_indices): 
                    factorToVar_edge_index.append([factor_idx, var_idx])
                    edge_var_indices.append([var_idx_in_factor, -99])
                factorToVar_double_list.append(var_indices)
                factor_idx += 1

                if len(all_factor_dimensions) == factor_count: #done with the preamble
                    preamble = False
                    assert(factor_idx == factor_count)
                    factor_idx = -1 #reset for processing potentials
                    continue

            else: #done with preamble, processing potentials
                split_line = line.split()
                if len(split_line) == 0:
                    continue
                elif factor_entry_count is None:
                    # assert(len(split_line) == 1)
                    factor_entry_count = int(split_line[0])
                    factor_entries = []
                    factor_idx += 1
                    if len(split_line) > 1:
                        factor_entries.extend([float(factor_entry) for factor_entry in split_line[1:]])
                        assert(len(factor_entries) <= factor_entry_count)
                        if len(factor_entries) == factor_entry_count: #store potential
                            factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in factorToVar_double_list[factor_idx]]
                            factor_potential = torch.tensor(factor_entries).reshape(factor_var_cardinalities)
                            factor_potentials.append(factor_potential)
                            factor_entries = None
                            factor_entry_count = None                    
                else:
                    factor_entries.extend([float(factor_entry) for factor_entry in split_line])
                    assert(len(factor_entries) <= factor_entry_count)
                    if len(factor_entries) == factor_entry_count: #store potential
                        factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in factorToVar_double_list[factor_idx]]
                        factor_potential = torch.tensor(factor_entries).reshape(factor_var_cardinalities)
                        factor_potentials.append(factor_potential)
                        factor_entries = None
                        factor_entry_count = None
    assert(factor_entries is None)
    assert(factor_entry_count is None)
    assert(factor_idx + 1 == factor_count)
    assert(len(factor_potentials) == factor_count)

    #check if all variables have the same cardinality
    all_variables_share_cardinality = True
    var_cardinality = variable_cardinalities[0]
    for var_card in variable_cardinalities:
        if var_card != var_cardinality:
            all_variables_share_cardinality = False
            break
            # return False #haven't implemented variables with different cardinality
    if all_variables_share_cardinality and RUN_LBP:
        libDAI_ln_z_estimate = run_loopyBP(factor_potentials, factorToVar_double_list, N=variable_count,\
            var_cardinality=var_cardinality, maxiter=1000, updates="SEQRND", damping=None)

        with open(problem_directory+'/train_test_split.json', 'r') as json_file:
            all_problems = json.load(json_file)
            partition_function = None
            for problem in all_problems['train_problems']:
                if problem["name"] == file_name:
                    partition_function = problem['exact_log_Z']
                    uai_bp_est = problem['bpEst_log_Z']
                    break
            if partition_function is None:
                for problem in all_problems['test_problems']:
                    if problem["name"] == file_name:
                        partition_function = problem['exact_log_Z']
                        uai_bp_est = problem['bpEst_log_Z']
                        break
            
        assert(partition_function is not None), "Error reading ground truth partition function"

        print("libDAI_ln_z_estimate:", libDAI_ln_z_estimate)
        print("uai_bp_est:", uai_bp_est*np.log(10))
        print("uai partition_function:", partition_function*np.log(10))


    return all_variables_share_cardinality, var_cardinality, max_factor_dimension

if __name__ == "__main__":
    small_factor_dimension_cutoff = 3


    # for problem_cat in ['ObjDetect', 'Segment', 'Grids', 'DBN', 'CSP', 'Protein', 'Promedas', 'BN', 'pedigree']:
    for problem_cat in ['Promedas', 'pedigree', 'ObjDetect', 'Segment', 'Grids', 'DBN', 'CSP', 'Protein']:
    # for problem_cat, problems in test_grouped_by_category.items():
        #problems we can't use because the don't all have the same variable cardinality
        different_variable_cardinalities = []
        #problems with a maximum factor dimension less than or equal to small_factor_dimension_cutoff
        #dictionary with:
        # key: variable cardinality
        # value: list of problem names
        small_facDim_byCardinality = defaultdict(list)
        #dictionary with:
        # key: variable cardinality
        # value: list of maximum factor dimensions by problems
        small_facDim_listMaxDim = defaultdict(list)

        #problems with a maximum factor dimension greater than small_factor_dimension_cutoff
        #dictionary with:
        # key: variable cardinality
        # value: list of problem names
        large_facDim_byCardinality = defaultdict(list)
        #dictionary with:
        # key: variable cardinality
        # value: list of maximum factor dimensions by problems
        large_facDim_listMaxDim = defaultdict(list)

        print("problem_cat:", problem_cat)
        with open('./data/'+problem_cat+'/train_test_split.json', 'r') as json_file:
            problem_names = json.load(json_file)
        problems = problem_names['train_problems']
        successfully_processed_count = 0
        un_successfully_processed_count = 0
        for problem in problems:
            # print("processing problem:", problem['name'])
            if not path.exists('./data/'+problem_cat+'/'+problem['name']):
                # print("missing", problem)
                continue
            all_variables_share_cardinality, var_cardinality, max_factor_dimension = inspect_uai_model(file_name=problem['name'],\
                                                                                                       problem_directory='./data/'+problem_cat+'/')
            if not all_variables_share_cardinality:
                different_variable_cardinalities.append(problem)
            elif max_factor_dimension <= small_factor_dimension_cutoff:
                small_facDim_byCardinality[var_cardinality].append(problem)
                small_facDim_listMaxDim[var_cardinality].append(max_factor_dimension)
            else:
                large_facDim_byCardinality[var_cardinality].append(problem)
                large_facDim_listMaxDim[var_cardinality].append(max_factor_dimension)

        print('-'*80)
        print("problem_cat:", problem_cat)
        print(len(different_variable_cardinalities), "problems with variables of different cardinalities:", different_variable_cardinalities)
        print()
        print("problems with a maximum factor dimension of", small_factor_dimension_cutoff)
        for var_cardinality, problems in small_facDim_byCardinality.items():
            print(len(problems), "with variable cardinality =", var_cardinality, "maximum factor dimensionality =", max(small_facDim_listMaxDim[var_cardinality]))
            # print(problems)
        print()
        print("problems with a maximum factor dimension greater than", small_factor_dimension_cutoff)
        for var_cardinality, problems in large_facDim_byCardinality.items():
            print(len(problems), "with variable cardinality =", var_cardinality, "maximum factor dimensionality =", max(large_facDim_listMaxDim[var_cardinality]))
            # print(problems)
        print()

    exit(0)

    




'''
#Data info
problem_cat: Pedigree
successfully_processed_count: 0
un_successfully_processed_count: 2

problem_cat: prob0
max_factor_dimension: 5 var_cardinality: 2
max_factor_dimension: 6 var_cardinality: 2
max_factor_dimension: 5 var_cardinality: 2
max_factor_dimension: 6 var_cardinality: 2
successfully_processed_count: 4
un_successfully_processed_count: 0

problem_cat: 2bit
max_factor_dimension: 11 var_cardinality: 2
max_factor_dimension: 6 var_cardinality: 2
successfully_processed_count: 2
un_successfully_processed_count: 0

problem_cat: linkage
successfully_processed_count: 0
un_successfully_processed_count: 11

problem_cat: relational
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
successfully_processed_count: 3
un_successfully_processed_count: 0

problem_cat: BN
successfully_processed_count: 0
un_successfully_processed_count: 5

problem_cat: logistics
max_factor_dimension: 14 var_cardinality: 2
max_factor_dimension: 14 var_cardinality: 2
successfully_processed_count: 2
un_successfully_processed_count: 0

problem_cat: ObjectDetection
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 21
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 11
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 16
max_factor_dimension: 2 var_cardinality: 11
successfully_processed_count: 45
un_successfully_processed_count: 0

problem_cat: r_problems
max_factor_dimension: 9 var_cardinality: 2
max_factor_dimension: 9 var_cardinality: 2
successfully_processed_count: 2
un_successfully_processed_count: 0

problem_cat: DBN
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
successfully_processed_count: 4
un_successfully_processed_count: 0

problem_cat: ungrouped
max_factor_dimension: 4 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
successfully_processed_count: 2
un_successfully_processed_count: 0

problem_cat: CSP
successfully_processed_count: 0
un_successfully_processed_count: 2

problem_cat: Grids
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
successfully_processed_count: 5
un_successfully_processed_count: 0

problem_cat: c_problems
max_factor_dimension: 9 var_cardinality: 2
max_factor_dimension: 10 var_cardinality: 2
successfully_processed_count: 2
un_successfully_processed_count: 0

problem_cat: Promedus
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
max_factor_dimension: 3 var_cardinality: 2
successfully_processed_count: 19
un_successfully_processed_count: 0

problem_cat: Segmentation
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
max_factor_dimension: 2 var_cardinality: 2
successfully_processed_count: 4
un_successfully_processed_count: 0
'''    