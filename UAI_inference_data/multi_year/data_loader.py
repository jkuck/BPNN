import random
import numpy as np
import torch
from factor_graph import FactorGraphData
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
import json
from os import path

# from libDAI_utils import run_loopyBP

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
problem_cat: DBN LBP error = 4.618604028
problems with a maximum factor dimension of 3
47 with variable cardinality = 2 maximum factor dimensionality = 2

--------------------------------------------------------------------------------
problem_cat: Promedas LBP error = 0.2545449077
problems with a maximum factor dimension of 3
46 with variable cardinality = 2 maximum factor dimensionality = 3

MAYBE CAN USE, SMALL DATASET
--------------------------------------------------------------------------------
problem_cat: Grids LBP error = 58.34310638
problems with a maximum factor dimension of 3
6 with variable cardinality = 2 maximum factor dimensionality = 2


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
                      dataset_type='train', evidence_file=None, partition_function=None, nEvid=None):
    '''
    Inputs:
    - max_allowed_factor_dimension (int): do not construct a factor graph if the largest factor has more than this many dimensions.
        set factor dimensionality to this value
    - dataset_type (string): train/test

    Outputs:
    - factor_graph (FactorGraphData): factor graph representation, or None if all variables do not have the same cardinality
    '''
    # format info here: http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html

    if dataset_type is None:
        # when creating extra data, set dataset_type=None, but specify partition_function and nEvid
        assert(partition_function is not None)
        assert(nEvid is not None)
    else:
        with open(problem_directory+'/train_test_split.json', 'r') as json_file:
            all_problems = json.load(json_file)
            if dataset_type == 'eval2012':
                problems = all_problems['train_problems']
                problems.extend(all_problems['test_problems'])
            elif dataset_type == 'train' or dataset_type == 'debug_train':
                problems = all_problems['train_problems']
            else:
                assert(dataset_type == 'test')
                problems = all_problems['test_problems']
            partition_function = None
            for problem in problems:
                if problem["name"] == file_name:
                    partition_function = problem['exact_log_Z']
                    nEvid = problem['nEvid']
                    break
            
    assert(partition_function is not None), "Error reading ground truth partition function"

    factor_potentials, factorToVar_double_list, new_variable_count, non_evidence_variable_cardinalities, factorToVar_edge_index, edge_var_indices =\
        process_helper(file_name, problem_directory, nEvid, evidence_file=evidence_file)

    #check if all variables have the same cardinality
    var_cardinality = non_evidence_variable_cardinalities[0]
    for var_card in non_evidence_variable_cardinalities: #it's ok if a variable in the evidence has a different cardinality since we condition on it and remove it
        if var_card != var_cardinality:
            return None #haven't implemented variables with different cardinality

    max_factor_dimension = 0
    for factor_potential in factor_potentials:
        if len(factor_potential.shape) > max_factor_dimension:
            max_factor_dimension = len(factor_potential.shape)

    if max_allowed_factor_dimension is not None:
        if max_factor_dimension > max_allowed_factor_dimension:
            return None #this factor graph has factor with more dimensions than is allowed
        else:
            max_factor_dimension = max_allowed_factor_dimension #construct the factor graph representing every factor with max_allowed_factor_dimension dimensions
    # print("max_factor_dimension:", max_factor_dimension, "var_cardinality:", var_cardinality)
    # return True

    padded_factor_potentials = [] #pad factor potentials so they all have the same size
    factor_potential_masks = [] #store masks: 1 signifies an invalid location (padding), 0 signifies a valid location
    for factor_idx, factor_potential in enumerate(factor_potentials):
        # if all_factor_dimensions[factor_idx] == max_factor_dimension:
        if len(factor_potential.shape) == max_factor_dimension:
            padded_factor_potentials.append(factor_potential)
            factor_potential_masks.append(torch.zeros([var_cardinality for i in range(max_factor_dimension)]))
        else:
            padded_size = [var_cardinality for i in range(max_factor_dimension)]
            padded_factor_potential, mask = build_padded_factor_potential(padded_size=padded_size, factor_potential=factor_potential)
            padded_factor_potentials.append(padded_factor_potential)
            factor_potential_masks.append(mask)

    # print("file_name:", file_name)

    # sleep(debug_quick)

    factor_count = len(padded_factor_potentials)
    factor_potentials = torch.stack(padded_factor_potentials, dim=0)
    factor_potential_masks = torch.stack(factor_potential_masks, dim=0)
    factorToVar_edge_index = torch.tensor(factorToVar_edge_index).t().contiguous()
    edge_var_indices = torch.tensor(edge_var_indices).t().contiguous()
    ln_Z = np.log(10)*partition_function #UAI stores partition function in log base 10, convert to log base e
    # assert((factor_potentials <= 1).all()), (file_name)
    ln_factor_potentials = torch.log(factor_potentials)
    # assert((ln_factor_potentials <= 0).all())

    factor_graph = FactorGraphData(factor_potentials=ln_factor_potentials,
                 factorToVar_edge_index=factorToVar_edge_index, numVars=new_variable_count, numFactors=factor_count, 
                 edge_var_indices=edge_var_indices, state_dimensions=max_factor_dimension,
                 factor_potential_masks=factor_potential_masks, ln_Z=ln_Z, factorToVar_double_list=factorToVar_double_list,
                 var_cardinality=var_cardinality, belief_repeats=belief_repeats)
    return factor_graph




class UAI_Dataset(InMemoryDataset):    
    def __init__(self, root, dataset_type, problem_category, belief_repeats, max_factor_dimension, var_cardinality,\
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
        self.var_cardinality = var_cardinality
        self.PROBLEMS_DIR = root + problem_category + "/"

        assert(self.dataset_type in ['train', 'test', 'debug_train', 'eval2012'])
        # if problem_category[:5] == 'group':
        #     self.max_factor_dimension = max_factor_dimension_by_groups[problem_category]
        #     if self.dataset_type == 'test':
        #         self.problem_names = test_groups[problem_category]
        #     else:
        #         self.problem_names = train_groups[problem_category]

        if dataset_type == 'eval2012':
            assert(problem_category == 'DBN')
            self.problem_names = ['rbm_20.uai', 'rbm_21.uai', 'rbm_22.uai', 'rbm_ferro_20.uai', 'rbm_ferro_21.uai', 'rbm_ferro_22.uai']

        else:
            with open(self.PROBLEMS_DIR +'train_test_split.json', 'r') as json_file:
                all_problems = json.load(json_file)
                if dataset_type == 'train' or dataset_type == 'debug_train':
                    problems = all_problems['train_problems']
                else:
                    assert(dataset_type == 'test')
                    problems = all_problems['test_problems']

                if dataset_type == 'debug_train':
                    self.problem_names = [problem["name"] for problem in problems[:1]]
                    print("self.problem_names:", self.problem_names)
                else:
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
        processed_dataset_file = self.dataset_type + '_%s_%d_%d_%d_pyTorchGeomProccesed.pt' %\
                                 (self.problem_category, self.belief_repeats, self.max_factor_dimension, self.var_cardinality)
        
        return [processed_dataset_file]

    def download(self):
        pass
        # assert(False), "Error, need to generate new data!!"
        # Download to `self.raw_dir`.

    def process(self):
        # get a list of sat problems as factor graphs (FactorGraphData objects)    
        print("self.problem_names:", self.problem_names)
        problem_list = []
        for problem_name in self.problem_names:
            cur_factor_graph = process_uai_model(file_name=problem_name,\
                                                 problem_directory=self.PROBLEMS_DIR,\
                                                 belief_repeats=self.belief_repeats,\
                                                 max_allowed_factor_dimension=self.max_factor_dimension,
                                                 dataset_type=self.dataset_type)

            #make sure problem loaded and it's variable cardinality matches other problems we are loading                                                 
            if (cur_factor_graph is not None) and (cur_factor_graph.var_cardinality == self.var_cardinality):
                problem_list.append(cur_factor_graph)

        data, slices = self.collate(problem_list)
        torch.save((data, slices), self.processed_paths[0])

def condition_on_evidence(factor_potential, evidence_dict, variable_list):
    condition_factor_on_evidence = False
    for var_idx in variable_list:
        if var_idx in evidence_dict:
            condition_factor_on_evidence = True
            
    if condition_factor_on_evidence: #this factor contains a variable that is in the evidence
        for indices in np.ndindex(factor_potential.shape): #set all values in the factor to 0 that disagree with the evidence
            set_to_0 = False
            for dimension, index_val in enumerate(indices):
                var_idx = variable_list[dimension]
                if (var_idx in evidence_dict) and (index_val != evidence_dict[var_idx]):
                    set_to_0 = True
                    break
            if set_to_0:
                factor_potential[indices] = 0
    return factor_potential

def condition_on_evidence2(factor_potential, evidence_dict, variable_list, new_var_indices):
    # print("condition_on_evidence2 called :)")
    condition_factor_on_evidence = False
    new_variable_list = []
    # print("factor_potential:", factor_potential)
    for local_var_idx, global_var_idx in enumerate(variable_list):
        # local_var_idx is the index of the variable within the factor
        # global_var_idx is the index of the variable among all variables
        if global_var_idx in evidence_dict:
            var_val = evidence_dict[global_var_idx]
            factor_potential = torch.narrow(factor_potential, local_var_idx, var_val, 1)
            # print("narrow called")
        else:
            new_variable_list.append(new_var_indices[global_var_idx])        
    factor_potential = factor_potential.squeeze()
    if len(new_variable_list) == 0:
        assert(factor_potential.numel() == 1)
        scaling_factor = factor_potential.item()
        # print("scale by:", scaling_factor)
        # sleep(asd)
        factor_potential = None
        new_variable_list = None
    else:
        scaling_factor = None
    return factor_potential, new_variable_list, scaling_factor

def process_helper(file_name, problem_directory, nEvid, evidence_file):
    evidence_dict = {}
    if nEvid != 0: 
        #process evidence file
        if evidence_file is None:
            evidence_file = problem_directory + 'evidence/' + file_name + '.evid'
        assert(path.isfile(evidence_file))
        with open(evidence_file, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.split()
                evidence_var_count = int(line[0])
                evidence = line[1:]
                assert(len(evidence) == 2*evidence_var_count)
                for idx in range(0, len(evidence), 2):
                    var_idx = int(evidence[idx]) #the index of the variable
                    var_val = int(evidence[idx+1]) #the value it is set to
                    evidence_dict[var_idx] = var_val

                break #the info should be on the first line of the file
        assert(len(evidence_dict) == nEvid), (len(evidence_dict), nEvid)


    with open(problem_directory + file_name, 'r') as f:
        preamble = True
        factor_entries = None
        factor_entry_count = None

        factor_idx = 0
        factorToVar_edge_index = []
        edge_var_indices = []
        # factorToVar_double_list[i][j] is the index of the jth variable in the ith factor 
        factorToVar_double_list = []
        new_factorToVar_double_list = [] #after conditioning on evidence

        factor_potentials = []
        
        all_factor_dimensions = [] #all_factor_dimensions[i] is the number of variables in the ith factor

        total_scaling_factor = 1.0
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                assert(line.split()[0] == "MARKOV"), line
                continue
            if line_idx == 1:
                variable_count = int(line)
                if nEvid != 0:
                    for var_idx, var_value in evidence_dict.items():
                        assert(var_idx >= 0 and var_idx < variable_count), (var_idx, variable_count)
                    new_var_indices = {}
                    new_var_idx = 0
                    for old_var_idx in range(variable_count):
                        if old_var_idx not in evidence_dict:
                            new_var_indices[old_var_idx] = new_var_idx
                            new_var_idx += 1
                        # else:
                        #     new_var_indices[old_var_idx] = None #variable is removed
                    assert(len(new_var_indices) == variable_count - nEvid), (len(new_var_indices), variable_count - nEvid, variable_count, nEvid, evidence_dict, new_var_idx)
                    new_variable_count = len(new_var_indices)
                else:
                    new_variable_count = variable_count

                continue
            if line_idx == 2:
                variable_cardinalities = line.split()
                #var_cardinalities[i] is the cardinality of the ith variable
                variable_cardinalities = [int(var_card) for var_card in variable_cardinalities]
                non_evidence_variable_cardinalities = [int(var_card) for (var_idx, var_card) in enumerate(variable_cardinalities) if (var_idx not in evidence_dict)]
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
                var_indices = [int(var_idx) for var_idx in split_line[1:]] #indices of all variables in this factor
                assert(len(var_indices) == factor_dimension)
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
                            if nEvid != 0:
                                # factor_potential = condition_on_evidence(factor_potential, evidence_dict, variable_list=factorToVar_double_list[factor_idx])
                                factor_potential, new_variable_list, scaling_factor = condition_on_evidence2(factor_potential, evidence_dict, variable_list=factorToVar_double_list[factor_idx], new_var_indices=new_var_indices)
                                if scaling_factor is not None:
                                    total_scaling_factor *= scaling_factor
                                else:
                                    #enumerate the variable indices within the factor and the global variable indices
                                    for var_idx_in_factor, var_idx in enumerate(new_variable_list): 
                                        factorToVar_edge_index.append([factor_idx, var_idx])
                                        edge_var_indices.append([var_idx_in_factor, -99])

                                    assert(len(new_factorToVar_double_list) == len(factor_potentials)), (len(new_factorToVar_double_list), len(factor_potentials))
                                    new_factorToVar_double_list.append(new_variable_list)
                                    factor_potentials.append(factor_potential)
                                    assert(len(new_factorToVar_double_list) == len(factor_potentials)), (len(new_factorToVar_double_list), len(factor_potentials))
                            else:
                                factor_potentials.append(factor_potential)
                                #enumerate the variable indices within the factor and the global variable indices
                                for var_idx_in_factor, var_idx in enumerate(factorToVar_double_list[factor_idx]): 
                                    factorToVar_edge_index.append([factor_idx, var_idx])
                                    edge_var_indices.append([var_idx_in_factor, -99])

                            factor_entries = None
                            factor_entry_count = None                    
                else:
                    factor_entries.extend([float(factor_entry) for factor_entry in split_line])
                    assert(len(factor_entries) <= factor_entry_count)
                    if len(factor_entries) == factor_entry_count: #store potential
                        factor_var_cardinalities = [variable_cardinalities[var_idx] for var_idx in factorToVar_double_list[factor_idx]]
                        factor_potential = torch.tensor(factor_entries).reshape(factor_var_cardinalities)
                        if nEvid != 0:
                            # factor_potential = condition_on_evidence(factor_potential, evidence_dict, variable_list=factorToVar_double_list[factor_idx])
                            factor_potential, new_variable_list, scaling_factor = condition_on_evidence2(factor_potential, evidence_dict, variable_list=factorToVar_double_list[factor_idx], new_var_indices=new_var_indices)
                            if scaling_factor is not None:
                                total_scaling_factor *= scaling_factor
                            else:
                                #enumerate the variable indices within the factor and the global variable indices
                                for var_idx_in_factor, var_idx in enumerate(new_variable_list): 
                                    factorToVar_edge_index.append([factor_idx, var_idx])
                                    edge_var_indices.append([var_idx_in_factor, -99])

                                assert(len(new_factorToVar_double_list) == len(factor_potentials)), (len(new_factorToVar_double_list), len(factor_potentials))
                                new_factorToVar_double_list.append(new_variable_list)
                                factor_potentials.append(factor_potential)
                                assert(len(new_factorToVar_double_list) == len(factor_potentials)), (len(new_factorToVar_double_list), len(factor_potentials))
                        else:
                            factor_potentials.append(factor_potential)
                            #enumerate the variable indices within the factor and the global variable indices
                            for var_idx_in_factor, var_idx in enumerate(factorToVar_double_list[factor_idx]): 
                                factorToVar_edge_index.append([factor_idx, var_idx])
                                edge_var_indices.append([var_idx_in_factor, -99])

                        factor_entries = None
                        factor_entry_count = None
    #rescale the first factor (arbitrary) by the single values of all factors ended up with only one state after conditioning on the evidence    
    print("RESCALE by:", total_scaling_factor)                    
    factor_potentials[0] = total_scaling_factor * factor_potentials[0]
    if nEvid != 0:
        factorToVar_double_list = new_factorToVar_double_list
    assert(len(factorToVar_double_list) == len(factor_potentials)), (len(factorToVar_double_list), len(factor_potentials))
    assert(factor_entries is None)
    assert(factor_entry_count is None)
    assert(factor_idx + 1 == factor_count)
    if nEvid == 0:
        assert(len(factor_potentials) == factor_count)
    
    return factor_potentials, factorToVar_double_list, new_variable_count, non_evidence_variable_cardinalities, factorToVar_edge_index, edge_var_indices

def inspect_uai_model(file_name='BN_0.uai',\
                      problem_directory='./data/BN/',\
                      belief_repeats=1, RUN_LBP=True, evidence_file=None):
    '''
    Inpspect the specified UAI model

    Outputs:
    - variables_equal_cardinality (bool): True if the variables all have the same cardinality
    - var_cardinality (int): if variables_equal_cardinality=True, the cardinality of all variables
    - max_factor_dimension
    '''
    # format info here: http://www.hlt.utdallas.edu/~vgogate/uai14-competition/modelformat.html
    with open(problem_directory+'/train_test_split.json', 'r') as json_file:
        all_problems = json.load(json_file)
        partition_function = None
        for problem in all_problems['train_problems']:
            if problem["name"] == file_name:
                partition_function = problem['exact_log_Z']
                uai_bp_est = problem['bpEst_log_Z']
                nEvid = problem['nEvid']
                break
        if partition_function is None:
            for problem in all_problems['test_problems']:
                if problem["name"] == file_name:
                    partition_function = problem['exact_log_Z']
                    uai_bp_est = problem['bpEst_log_Z']
                    nEvid = problem['nEvid']
                    break
        
    assert(partition_function is not None), "Error reading ground truth partition function"

    factor_potentials, factorToVar_double_list, new_variable_count, non_evidence_variable_cardinalities, factorToVar_edge_index, edge_var_indices = process_helper(file_name, problem_directory, nEvid, evidence_file)

    #check if all variables have the same cardinality
    all_variables_share_cardinality = True
    var_cardinality = non_evidence_variable_cardinalities[0]
    for var_card in non_evidence_variable_cardinalities: #it's ok if a variable in the evidence has a different cardinality since we condition on it and remove it
        if var_card != var_cardinality:
            all_variables_share_cardinality = False
            break
            # return False #haven't implemented variables with different cardinality
            
    max_factor_dimension = 0
    for factor_potential in factor_potentials:
        if len(factor_potential.shape) > max_factor_dimension:
            max_factor_dimension = len(factor_potential.shape)


    if all_variables_share_cardinality and RUN_LBP:
        libDAI_ln_z_estimate = run_loopyBP(factor_potentials, factorToVar_double_list, N=new_variable_count,\
            var_cardinality=var_cardinality, maxiter=1000, updates="PARALL", damping='.5')
            # var_cardinality=var_cardinality, maxiter=1000, updates="SEQRND", damping='.5')
            # var_cardinality=var_cardinality, maxiter=1000, updates="SEQFIX", damping=None)


        print("libDAI_ln_z_estimate:", libDAI_ln_z_estimate)
        print("uai_bp_est:", uai_bp_est*np.log(10))
        print("uai partition_function:", partition_function*np.log(10))

    squared_error = (partition_function*np.log(10) - libDAI_ln_z_estimate)**2
    uai_squared_error = (partition_function*np.log(10) - uai_bp_est*np.log(10))**2
    return all_variables_share_cardinality, var_cardinality, max_factor_dimension, squared_error, uai_squared_error

if __name__ == "__main__":
    small_factor_dimension_cutoff = 3


    # for problem_cat in ['ObjDetect', 'Segment', 'Grids', 'DBN', 'CSP', 'Protein', 'Promedas', 'BN', 'pedigree']:
  
    # for problem_cat in ['Promedas', 'pedigree', 'ObjDetect', 'Segment', 'Grids', 'DBN', 'CSP', 'Protein']:
    # for problem_cat in ['Promedas', 'ObjDetect', 'DBN', 'Segment', 'pedigree', 'Grids', 'CSP', 'Protein']:
    for problem_cat in ['DBN']:
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
        squared_errors = []
        UAI_squared_errors = []
        for problem in problems:
            # if problem['name'] != 'giraffe_rescaled_5004.K10.F1.75.model.uai':
            #     continue
            
            print()
            print("processing problem:", problem['name'])
            if not path.exists('./data/'+problem_cat+'/'+problem['name']):
                # print("missing", problem)
                continue
            all_variables_share_cardinality, var_cardinality, max_factor_dimension, squared_error, uai_squared_error = inspect_uai_model(file_name=problem['name'],\
                                                                                                       problem_directory='./data/'+problem_cat+'/')
            squared_errors.append(squared_error)
            UAI_squared_errors.append(uai_squared_error)
            if not all_variables_share_cardinality:
                different_variable_cardinalities.append(problem)
            elif max_factor_dimension <= small_factor_dimension_cutoff:
                small_facDim_byCardinality[var_cardinality].append(problem)
                small_facDim_listMaxDim[var_cardinality].append(max_factor_dimension)
            else:
                large_facDim_byCardinality[var_cardinality].append(problem)
                large_facDim_listMaxDim[var_cardinality].append(max_factor_dimension)

            print("done processing problem:", problem['name'])

        print("libdai RMSE =", np.sqrt(np.mean(squared_errors)))
        print("UAI RMSE =", np.sqrt(np.mean(UAI_squared_errors)))
        print("UAI median error =", np.sqrt(np.median(UAI_squared_errors)))
        print("UAI_squared_errors:", UAI_squared_errors)
        sleep(RMSE)

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

    

