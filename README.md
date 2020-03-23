# learn_BP

## Integrating SBM todo's
- finish StochasticBlockModel (see "add code here for sampling node labels") in community_detection/sbm_data.py
- make sure wandb is set up.  this is helpful for looking at experimental results and is how i save models as well.
- finish learn_BP/learn_BP_communityDetection.py 
	- edit the loss function
	- other minor changes
- make a network that doesn't have a Bethe final layer, instead just apply loss to final variable beliefs and gt node labels

## Dependencies
[mrftools](https://bitbucket.org/berthuang/mrftools/src/master/) (not required, can comment out imports)
[https://github.com/dbtsai/libDAI](libDAI) used with Ising model experiments:
- junction tree algorithm for obtaining the exact partition function
- loopy belief propagation for comparison

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


## SAT data info
- data/sat_problems_noIndSets contains SAT problems stripped of sampling sets and independent sets
- data/sat_problems_IndSetsRecomputed contains SAT problems stripped of sampling sets and with independent sets recomputed with a timeout of 1000 seconds

