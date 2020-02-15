# learn_BP


## Training the BPNN on Ising models
Run learn_BP_spinGlass.py with mode="train"

## Training the GNN
Run learn_GNN_spinGlass.py with mode="train"

## Making figures
Run learn_BP_spinGlass.py with mode="test" 
- To make the table call create_many_ising_model_figures() 
- To make a single figure, call create_ising_model_figure()
- Set BPNN_trained_model_path and GNN_trained_model_path to the appropriate trained model paths

## SAT data info
- data/sat_problems_noIndSets contains SAT problems stripped of sampling sets and independent sets
- data/sat_problems_IndSetsRecomputed contains SAT problems stripped of sampling sets and with independent sets recomputed with a timeout of 1000 seconds

