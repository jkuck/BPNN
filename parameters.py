ROOT_DIR = "/atlas/u/jkuck/learn_BP/" #file path to the directory cloned from github

alpha = 1 #residual weighting on messages, e.g. damping.  alpha=1 corresponds to no damping, alpha=0 is total damping
alpha2 = 0.99 #residual weighting directly around MLP. alpha2=1 corresponds to no neural network, alpha2=0 is no residual connection      
SHARE_WEIGHTS = False #if true, share the weights between layers in a BPNN
BETHE_MLP = True #if true, use a final layer MLP for BPNN that is initialized to the Bethe approximation
NUM_MLPS = 2 #the number of MLPs per layer in BPNN (1 or 2)

MRFTOOLS_LBP_ITERS = 5 #iterations of loopy belief propagation to run in mrftools
LIBDAI_LBP_ITERS = 5 #iterations of loopy belief propagation to run in libdai
LIBDAI_MEAN_FIELD_ITERS = 5 #iterations of mean field to run in libdai


# LN_ZERO = -np.inf
LN_ZERO = -99 #for numerical stability, use this value instead of -infinity for ln(0)