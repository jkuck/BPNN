import numpy as np
from . import mrftools_utils
from . import libdai_utils

class SpinGlassModel:
    def __init__(self, N, f, c, all_weights_1=False):
        '''
        Sample local field parameters and coupling parameters to define a spin glass model
        
        Inputs:
        - N: int, the model will be a grid with shape (NxN)
        - f: float, local field parameters (theta_i) will be drawn uniformly at random
            from [-f, f] for each node in the grid
        - c: float, coupling parameters (theta_ij) will be drawn uniformly at random from
            [0, c) (gumbel paper uses [0,c], but this shouldn't matter) for each edge in 
            the grid
        - all_weights_1: bool, if true return a model with all weights = 1 so Z=2^(N^2)
                               if false return randomly sampled model

        Values defining the spin glass model:
        - lcl_fld_params: array with dimensions (NxN), local field parameters (theta_i)
            that we sampled for each node in the grid 
        - cpl_params_h: array with dimensions (N x N-1), coupling parameters (theta_ij)
            for each horizontal edge in the grid.  cpl_params_h[k,l] corresponds to 
            theta_ij where i is the node indexed by (k,l) and j is the node indexed by
            (k,l+1)
        - cpl_params_v: array with dimensions (N-1 x N), coupling parameters (theta_ij)
            for each vertical edge in the grid.  cpl_params_h[k,l] corresponds to 
            theta_ij where i is the node indexed by (k,l) and j is the node indexed by
            (k+1,l)     
        '''
        self.N = N

        if all_weights_1: #make all weights 1
            #sample local field parameters (theta_i) for each node
            self.lcl_fld_params = np.zeros((N,N))
    
            #sample horizontal coupling parameters (theta_ij) for each horizontal edge
            self.cpl_params_h = np.zeros((N,N-1))
    
            #sample vertical coupling parameters (theta_ij) for each vertical edge
            self.cpl_params_v = np.zeros((N-1,N))

        else: #randomly sample weights
            #sample local field parameters (theta_i) for each node
            self.lcl_fld_params = np.random.uniform(low=-f, high=f, size=(N,N))
            # self.lcl_fld_params = np.random.uniform(low=.5, high=f, size=(N,N))
            # lcl_fld_params_pos = np.random.uniform(low=1.5, high=f, size=(N,N))
            # randomarray = np.random.uniform(low=0, high=1, size=(N,N))
            # self.lcl_fld_params[np.where(randomarray>.5)] = lcl_fld_params_pos[np.where(randomarray>.5)]

            #sample horizontal coupling parameters (theta_ij) for each horizontal edge
            self.cpl_params_h = np.random.uniform(low=0.0, high=c, size=(N,N-1))
    
            #sample vertical coupling parameters (theta_ij) for each vertical edge
            self.cpl_params_v = np.random.uniform(low=0.0, high=c, size=(N-1,N))


    def brute_force_z_mrftools(self):
        '''
        Brute force calculate the partition function of this spin glass model using mrftools

        Outputs:
        - exact_z: the exact partition function
        '''
        exact_z = mrftools_utils.brute_force(self)
        return exact_z

    def loopyBP_mrftools(self):
        ln_Z_estimate = mrftools_utils.run_LBP(self)
        return ln_Z_estimate

    def junction_tree_libdai(self):
        '''
        Compute the partition function of this spin glass model using the junction tree 
        implementation from libdai

        Outputs:
        - ln_Z: natural logarithm of the exact partition function
        '''
        ln_Z = libdai_utils.junction_tree(self)
        return ln_Z

    def loopyBP_libdai(self):
        '''
        estimate the partition function of this spin glass model using the 
        loopy belief propagation implementation from libdai

        Outputs:
        - ln_Z_estimate: estimate of the natural logarithm of the partition function
        '''
        ln_Z_estimate = libdai_utils.run_loopyBP(self)
        return ln_Z_estimate

def compare_libdai_mrftools():
    sg_model = SpinGlassModel(N=10, f=1, c=1)
    print("exact ln(partition function):", sg_model.junction_tree_libdai())
    Z = sg_model.brute_force_z_mrftools()
    print("exact partition function:", Z, "ln(partition function):", np.log(Z))


if __name__ == "__main__":
    # compare_libdai_mrftools()
    # f = np.random.rand()*3
    # c = np.random.rand()*3
    f = 2
    c = 2
    sg_model = SpinGlassModel(N=10, f=f, c=c)
    print("f:", f, "c:", c, "exact ln(partition function):", sg_model.junction_tree_libdai())


