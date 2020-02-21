# learn_BP


## Training the BPNN on Ising models
Run learn_BP_spinGlass.py with mode="train"

## Training the GNN
Run learn_GNN_spinGlass.py with mode="train"

## Making figures/tables
Run learn_BP_spinGlass.py with mode="test" 
- To make the table call create_many_ising_model_figures() 
- To make a single figure, call create_ising_model_figure()
- Set BPNN_trained_model_path and GNN_trained_model_path to the appropriate trained model paths
- create latex table with data/experiments/make_latex_table.py

SAT table/figures
- Make the SAT tables in latex form from wandb results using data/wandbCSV_to_latexTable.py
- run data/compare_BPNNvsHashing_runtimes.py


