alpha = .5 #residual weighting on messages, e.g. damping.  alpha=1 corresponds to no skip connection, alpha=0 removes the neural network
alpha2 = .5 #residual weighting directly around MLP
MRFTOOLS_LBP_ITERS = 5 #iterations of loopy belief propagation to run in mrftools
LIBDAI_LBP_ITERS = 5 #iterations of loopy belief propagation to run in libdai
LIBDAI_MEAN_FIELD_ITERS = 5 #iterations of mean field to run in libdai