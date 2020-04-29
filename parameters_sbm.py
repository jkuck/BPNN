ROOT_DIR = "/sailhome/shuvamc/learn_BP" #file path to the directory cloned from github

alpha = .5 #residual weighting on messages, e.g. damping.  alpha=1 corresponds to no skip connection, alpha=0 removes the neural network
alpha2 = .4 #residual weighting directly around MLP
SHARE_WEIGHTS = False #if true, share the weights between layers in a BPNN
BETHE_MLP = False #if true, use a final layer MLP for BPNN that is initialized to the Bethe approximation
NUM_MLPS = 2 #the number of MLPs per layer in BPNN (1 or 2)

FINAL_MLP = True
N = 100
A=5
B=1
C=2
NUM_SAMPLES_TRAIN=10 
NUM_SAMPLES_VAL=1
SMOOTHING=False
BELIEF_REPEATS = 16
LN_ZERO = -99

MRFTOOLS_LBP_ITERS = 5 #iterations of loopy belief propagation to run in mrftools
LIBDAI_LBP_ITERS = 5 #iterations of loopy belief propagation to run in libdai
LIBDAI_MEAN_FIELD_ITERS = 5 #iterations of mean field to run in libdai
