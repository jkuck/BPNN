ROOT_DIR = "/sailhome/shuvamc/learn_BP" #file path to the directory cloned from github

alpha = .5 #residual weighting on messages, e.g. damping.  alpha=1 corresponds to no skip connection, alpha=0 removes the neural network
alpha2 = .4 #residual weighting directly around MLP
SHARE_WEIGHTS = False #if true, share the weights between layers in a BPNN
BETHE_MLP = True #if true, use a final layer MLP for BPNN that is initialized to the Bethe approximation
NUM_MLPS = 2 #the number of MLPs per layer in BPNN (1 or 2)

FINAL_MLP = False
LEARN_BP_INIT = False
NUM_BP_LAYERS = 15
PRE_BP_MLP = False
N = 20
A_TRAIN = 19
B_TRAIN = 1
C = 2
A_VAL = 19
B_VAL = 1
NUM_SAMPLES_TRAIN=5
NUM_SAMPLES_VAL=3
SMOOTHING=None
BELIEF_REPEATS = 1
LN_ZERO = -99

MRFTOOLS_LBP_ITERS = 5 #iterations of loopy belief propagation to run in mrftools
LIBDAI_LBP_ITERS = 5 #iterations of loopy belief propagation to run in libdai
LIBDAI_MEAN_FIELD_ITERS = 5 #iterations of mean field to run in libdai
