######### Stochastic Block Model Data #########
# This file implements generating community detection data sampled 
# from the stochastic block mode.  It also converts the data to
# a factor graph data format that is compatible with pytorch
# geometric (class FactorGraphData)



######### Sample SBM #########
#option 1, if you currently have a class representing the 
#stochastic block model then just move that here.
#this is equivalent to SpinGlassModel in learn_BP/ising_model/spin_glass_model.py
class StochasticBlockModel:
    def __init__(self):	
    	pass

#option 2, alternatively if you have a function that samples a stochastic block model
#and returns the representation as tensors or something other than a class, put that here
def sample_SBM():
	pass



######### Convert SBM to FactorGraphData #########
# convert your representation of a stochastic block model to FactorGraphData representation
# should do the same thing as build_factorgraph_from_SpinGlassModel
# in learn_BP/ising_model/pytorch_dataset.py
def build_factorgraph_from_sbm():
	pass




