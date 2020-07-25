#!/usr/bin/env python
# coding=utf-8

'''
Just load the files into the format of strings.
Those strings can be further preprocessed to fit the usage of libdai or torch_geometric
'''

import os, os.path as osp
import pickle as pkl

curdir = osp.dirname(osp.abspath(__file__))
datadir = osp.join(curdir, 'mpe')

def UAI2012Dataset(category=None):
    if category is None:
        filepath_list = os.listdir(datadir)
        filepath_list.remove('ProteinFolding')
        filepath_list.remove('CSP')
    else:
        filepath_list = [category]
    filepath_list = [osp.join(datadir, fp, 'uai', fn) for fp in filepath_list for fn in os.listdir(osp.join(datadir, fp, 'uai')) if fn[-4:]=='.uai']
    dataset, output_filepath_list = [], []
    for fn in filepath_list:
        with open(fn) as f:
            contents = f.readlines()
            contents = [c for c in contents if len(c.strip())]
        if 'MARKOV' in contents[0]:
            dataset.append(contents)
            output_filepath_list.append(fn)
    with open(osp.join(curdir,'BP_MAP_result.pkl'), 'rb') as f:
        best_bp_map_states_list = pkl.load(f)
    output_dataset = [(d,best_bp_map_states_list[p],p) for d,p in zip(dataset, output_filepath_list) if p in best_bp_map_states_list]
    return output_dataset

# codes for internal processing. The idx here is uniquely specified.
def _UAI2012Dataset(idx):
    filepath_list = os.listdir(datadir)
    filepath_list.remove('ProteinFolding')
    filepath_list.remove('CSP')
    filepath_list = [osp.join(datadir, fp, 'uai', fn) for fp in filepath_list for fn in os.listdir(osp.join(datadir, fp, 'uai')) if fn[-4:]=='.uai']

    filepath_list = [filepath_list[idx],]

    dataset, output_filepath_list = [], []
    for fn in filepath_list:
        with open(fn) as f:
            contents = f.readlines()
            contents = [c for c in contents if len(c.strip())]
        if 'MARKOV' in contents[0]:
            dataset.append(contents)
            output_filepath_list.append(fn)
    return dataset, output_filepath_list

if __name__ == '__main__':
    dataset = UAI2012Dataset()
    print(len(dataset), len(dataset[0]), len(dataset[1]))
