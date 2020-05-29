import numpy as np
import wandb
import torch
from community_detection.sbm_libdai import runLBPLibdai, runJT, build_libdaiFactorGraph_from_SBM
from community_detection.sbm_data_shuvam import StochasticBlockModel, build_factorgraph_from_sbm
from factor_graph import FactorGraphData
from factor_graph import DataLoader_custom as DataLoader_pytorchGeometric
from torch_geometric.data import Data, DataLoader, DataListLoader
import time
import os
import random
from nn_models_sbm import lbp_message_passing_network, GIN_Network_withEdgeFeatures 
from parameters_sbm_bethe import ROOT_DIR, alpha, alpha2, SHARE_WEIGHTS, FINAL_MLP, BETHE_MLP, EXACT_BETHE, NUM_MLPS, BELIEF_REPEATS, LEARN_BP_INIT, NUM_BP_LAYERS, PRE_BP_MLP, USE_MLP_1, USE_MLP_2, USE_MLP_3, USE_MLP_4, INITIALIZE_EXACT_BP, USE_MLP_DAMPING_FtoV, USE_MLP_DAMPING_VtoF

USE_WANDB = True
model_type = 'both'
gnn_model_path = '/atlas/u/shuvamc/model_weights/GINConv_moreweights_sbm_partition/epoch_300.pt'
bpnn_model_path = '/atlas/u/shuvamc/model_weights/bpnn_partition_mlp34_moredata_75prior/epoch_300.pt'  
N_TEST = 20
A_TEST = 19
B_TEST = 1
P_TEST = A_TEST / N_TEST
Q_TEST = B_TEST / N_TEST
C_TEST = 2
NUM_EXAMPLES_TEST = 5
gnn_numLayers = 30
gnn_featsize = 8
mode = 'marginals'
exp_name = mode + '_estimation_' + str(A_TEST) + '_' + str(B_TEST) + '_moredata'
if USE_WANDB:
    wandb.init(project="learn_partition_testing_sbm", name=exp_name)
    wandb.config.N_TEST = N_TEST
    wandb.config.P_TEST = P_TEST
    wandb.config.Q_TEST = Q_TEST
    wandb.config.C_TEST = C_TEST
    wandb.config.gnn_model_path = gnn_model_path
    wandb.config.bpnn_model_path = bpnn_model_path
    wandb.config.NUM_EXAMPLES = NUM_EXAMPLES_TEST * N_TEST * C_TEST

def calculatelogMarginals(partitions):
    partitions = torch.tensor(partitions)
    ln_sum = torch.logsumexp(partitions, dim = 0) 
    ln_marginals = partitions - ln_sum
    return ln_marginals.numpy()
    
 
def constructFixedDatasetBPNN(sbm_models_lst, init = True, debug = False, marginals = False, model = None):
    init = init or marginals
    fg_models = []
    init_mse = 0
    bp_marg_mse = 0
    mod_marg_mse = 0
    for sbm_model in sbm_models_lst:
        for i in range(N_TEST):
            jt_Z = []
            bp_Z = []
            fg_models_curr = []
            for j in range(C_TEST):
                sbm_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var = i, fixed_val = j)     
                jt_ln_z, jt_beliefs = runJT(sbm_fg, sbm_model)
                bpnn_fg = build_factorgraph_from_sbm(sbm_model, C_TEST, BELIEF_REPEATS, fixed_var = i, fixed_val = j, logZ = jt_ln_z)
                bpnn_fg.gt_variable_labels = jt_beliefs
                fg_models.append(bpnn_fg)
                if marginals:
                    fg_models_curr.append(bpnn_fg)
                if init:       
                    v_b, lbp_ln_z = runLBPLibdai(sbm_fg, sbm_model)
                    jt_Z.append(jt_ln_z)
                    bp_Z.append(lbp_ln_z)
                    print(jt_ln_z, lbp_ln_z)
                    #init_mse += (jt_ln_z - lbp_ln_z) ** 2
                    if debug:
                        model.to(torch.device('cpu'))
                        with torch.no_grad():
                            dl = DataLoader_pytorchGeometric([bpnn_fg], batch_size = 1)
                            for d in dl:
                                var_beliefs, est, fv_diff, vf_diff = model(d, torch.device('cpu'), perform_bethe = True)
                        print(jt_ln_z, lbp_ln_z, est)
                        time.sleep(2)
                        #print(fv_diff, vf_diff)
                        #print(v_b)
                        #time.sleep(5)
                        #print(torch.exp(var_beliefs))
                        #time.sleep(5)
                        #print(torch.max(torch.abs(var_beliefs - v_b)))
            if init:
                jt_Z = np.array(jt_Z)
                bp_Z = np.array(bp_Z)
                mse = min(np.sum((jt_Z-bp_Z)**2), np.sum((jt_Z - bp_Z[::-1])**2))
                print(mse)
                init_mse += mse
            if marginals:
                dl = DataLoader_pytorchGeometric(fg_models_curr, batch_size = C_TEST)
                for d in dl:
                    d.state_dimensions = d.state_dimensions[0] #hack for batching,
                    d.var_cardinality = d.var_cardinality[0] #hack for batching,
                    d.belief_repeats = d.belief_repeats[0]
                    with torch.no_grad():
                        _, preds, _, _ = model(d, torch.device('cpu'), perform_bethe = True)
                        preds = preds.squeeze(dim = 1)
                mod_marginals = calculatelogMarginals(preds)
                true_marginals = calculatelogMarginals(jt_Z)
                bp_marginals = calculatelogMarginals(bp_Z)
                mod_marg_mse += np.sum((true_marginals - mod_marginals) ** 2)#min(np.sum((true_marginals - mod_marginals)**2), np.sum((true_marginals - mod_marginals[::-1])**2))
                bp_marg_mse += min(np.sum((true_marginals - bp_marginals)**2), np.sum((true_marginals - bp_marginals[::-1])**2))
                print(preds, mod_marg_mse, bp_marg_mse)
    return fg_models, init_mse / len(fg_models), mod_marg_mse / len(fg_models), bp_marg_mse / len(fg_models)

  
def getDataPartitionGNN(num, p, q, classes, num_examples, prior_prob):
    normal_edge = np.array([p, q, q, p])
    normal_noedge = np.array([1-p, 1-q, 1-q, 1-p])
    normal_mask = np.ones(4)
    var_1_mask = normal_mask
    var_2_mask = normal_mask
    dataset = []
    init_mse = 0
    for i in range(num_examples):
        sbm_model = StochasticBlockModel(num, p, q, classes, community_probs = prior_prob)
        edge_index = sbm_model.edge_index_full.copy()
        noedge_index = np.concatenate((sbm_model.noedge_index, np.flip(sbm_model.noedge_index, axis = 0)), axis = 1)
        edge_attr = np.repeat(normal_edge[np.newaxis, :], edge_index.shape[1], axis = 0)
        noedge_attr = np.repeat(normal_noedge[np.newaxis, :], noedge_index.shape[1], axis = 0)
        for j in range(num):
            jt_Z_lst = []
            bp_Z_lst = []
            for k in range(classes):
                if k == 0:
                    var_1_mask[:2] = 0
                    var_2_mask[0::2] = 0
                else:
                    var_1_mask[-2:] = 0
                    var_2_mask[1::3] = 0
                edge_mask = np.array([var_1_mask if edge_index[1][m] == j else var_2_mask if edge_index[0][m] == j else normal_mask for m in range(edge_index.shape[1])])
                noedge_mask = np.array([var_1_mask if noedge_index[1][m] == j else var_2_mask if noedge_index[0][m] == j else normal_mask for m in range(noedge_index.shape[1])])
                edge_attr = edge_attr * edge_mask
                noedge_attr = noedge_attr * noedge_mask
                tot_edge_index = np.concatenate((edge_index, noedge_index), axis = 1)
                tot_edge_attr = np.concatenate((edge_attr, noedge_attr), axis = 0)
                libdai_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var = j, fixed_val = k)
                jt_Z, _ = runJT(libdai_fg, sbm_model)
                _, bp_Z = runLBPLibdai(libdai_fg, sbm_model)
                print(jt_Z, bp_Z)
                jt_Z_lst.append(jt_Z)
                bp_Z_lst.append(bp_Z)
                d = Data(x = torch.Tensor(sbm_model.deg_lst), y = torch.tensor([jt_Z]).float(), edge_index = torch.Tensor(tot_edge_index).long(), edge_attr = torch.Tensor(tot_edge_attr))
                dataset.append(d)
            jt_Z_lst = np.array(jt_Z_lst)
            bp_Z_lst = np.array(bp_Z_lst)
            mse = min(np.sum((jt_Z_lst-bp_Z_lst)**2), np.sum((jt_Z_lst - bp_Z_lst[::-1])**2))
            print(mse)
            init_mse += mse
    return dataset, init_mse / len(dataset)
 
def testBPNN(model, device, data, loss_func):
    tot_loss = 0
    for sbm_model in data:
        sbm_model.to(device)
        sbm_model.state_dimensions = sbm_model.state_dimensions[0] #hack for batching,
        sbm_model.var_cardinality = sbm_model.var_cardinality[0] #hack for batching,
        sbm_model.belief_repeats = sbm_model.belief_repeats[0]
        with torch.no_grad():
            _, estimated_partition, fv_diff, vf_diff = model(sbm_model, device, perform_bethe = True)
            estimated_partition = estimated_partition.squeeze(dim = 1)
            true_partition = sbm_model.ln_Z
            tot_loss += loss_func(estimated_partition, true_partition.float())
    return tot_loss / len(data)
        
def testGNN(loader, model, device, loss_func):
    tot_loss = 0
    for data in loader:
        data.to(device)
        edge_index = data.edge_index
        x = data.x
        edge_attr = data.edge_attr
        true_partition = data.y
        batch = data.batch
        with torch.no_grad():
            out = model(x, edge_index, edge_attr, batch)
            loss = loss_func(out, true_partition)
        tot_loss += loss.item()
    return tot_loss / len(loader)
        
if __name__ == '__main__':
    random.seed(10)
    np.random.seed(15)
    torch.manual_seed(20) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_func = torch.nn.MSELoss()
    if mode == 'partitions':
        if model_type == 'bpnn' or model_type == 'both':
            test_sbm_models_list = [StochasticBlockModel(N=N_TEST, P=P_TEST, Q=Q_TEST, C=C_TEST, community_probs = [.9, .1]) for i in range(NUM_EXAMPLES_TEST)]
            test_sbm_models_fg, libdai_test_mse, _, _ = constructFixedDatasetBPNN(test_sbm_models_list, init = (model_type == 'bpnn')) 
            test_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(test_sbm_models_fg, batch_size=16)
            model = lbp_message_passing_network(max_factor_state_dimensions=2, msg_passing_iters=30, device=device, share_weights = SHARE_WEIGHTS, bethe_MLP = BETHE_MLP, var_cardinality = C_TEST, belief_repeats = BELIEF_REPEATS, final_fc_layers = FINAL_MLP, learn_BP_init = LEARN_BP_INIT, num_BP_layers = NUM_BP_LAYERS, pre_BP_mlp = PRE_BP_MLP, use_mlp_1 = USE_MLP_1, use_mlp_2 = USE_MLP_2, use_mlp_3 = USE_MLP_3, use_mlp_4 = USE_MLP_4, init_exact_bp = INITIALIZE_EXACT_BP, mlp_damping_FtoV = USE_MLP_DAMPING_FtoV, mlp_damping_VtoF = USE_MLP_DAMPING_VtoF)
            model.load_state_dict(torch.load(bpnn_model_path))
            model.to(device) 
            bpnn_test_loss = testBPNN(model, device, test_data_loader_pytorchGeometric, loss_func)
        if model_type == 'gnn' or model_type == 'both':
            test_dataset, libdai_test_mse = getDataPartitionGNN(N_TEST, P_TEST, Q_TEST, C_TEST, NUM_EXAMPLES_TEST, prior_prob = [.9, .1])
            test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True)
            model = GIN_Network_withEdgeFeatures(input_state_size=1, edge_attr_size=2**C_TEST, hidden_size=gnn_featsize, msg_passing_iters=gnn_numLayers)
            model.load_state_dict(torch.load(gnn_model_path))
            model.to(device)
            gnn_test_loss = testGNN(test_loader, model, device, loss_func)
        print("Libdai Test MSE: " + str(libdai_test_mse))
        if USE_WANDB:
            wandb.log({"Libdai Test MSE": libdai_test_mse})
        if model_type == 'gnn' or model_type == 'both':
            print("GNN Test MSE: " + str(gnn_test_loss))
            if USE_WANDB:
                wandb.log({"GNN Test MSE": gnn_test_loss})
        if model_type == 'bpnn' or model_type == 'both':
            print("BPNN Test MSE: " + str(bpnn_test_loss))
            if USE_WANDB:
                wandb.log({"BPNN Test MSE": bpnn_test_loss})
    if mode == 'marginals':
        test_sbm_models_list = [StochasticBlockModel(N=N_TEST, P=P_TEST, Q=Q_TEST, C=C_TEST, community_probs = [.75, .25]) for i in range(NUM_EXAMPLES_TEST)]
        model = lbp_message_passing_network(max_factor_state_dimensions=2, msg_passing_iters=30, device=device, share_weights = SHARE_WEIGHTS, bethe_MLP = BETHE_MLP, var_cardinality = C_TEST, belief_repeats = BELIEF_REPEATS, final_fc_layers = FINAL_MLP, learn_BP_init = LEARN_BP_INIT, num_BP_layers = NUM_BP_LAYERS, pre_BP_mlp = PRE_BP_MLP, use_mlp_1 = USE_MLP_1, use_mlp_2 = USE_MLP_2, use_mlp_3 = USE_MLP_3, use_mlp_4 = USE_MLP_4, init_exact_bp = INITIALIZE_EXACT_BP, mlp_damping_FtoV = USE_MLP_DAMPING_FtoV, mlp_damping_VtoF = USE_MLP_DAMPING_VtoF)
        model.load_state_dict(torch.load(bpnn_model_path))
        _, _, mod_mse, bp_mse = constructFixedDatasetBPNN(test_sbm_models_list, init = True, marginals = True, model = model)
        print('Model Marginals MSE: ' + str(mod_mse))
        print("BP Marginals MSE: " + str(bp_mse))
        if USE_WANDB:
            wandb.log({"Model Marginals MSE": mod_mse, "BP Marginals MSE": bp_mse})
