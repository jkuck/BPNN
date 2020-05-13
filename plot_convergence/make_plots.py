import json
import matplotlib.pyplot as plt
import matplotlib

def make_plot(BPNN_data, BP_data, y_label = 'Max Difference', title='Factor to Variable Message Convergence'):
    num_runs = len(BPNN_data) #the number of problems we ran BPNN/BP on
    num_iterations = len(BPNN_data[0]) #the number of message passing iterations BPNN was run for
    print("num_iterations =", num_iterations)
    iteration_vals = [i for i in range(num_iterations)]
    for run_grouping in range(5):
        # for run_idx in range(num_runs):
        for run_idx in range(run_grouping*10, (run_grouping+1)*10):
            # print("BPNN_data[run_idx]:", BPNN_data[run_idx])
            plt.plot(iteration_vals, BPNN_data[run_idx], '--', color='b')#,label='BPNN')
            plt.plot(iteration_vals, BP_data[run_idx], '--', color='r')#, label='BP')

        plt.xlabel('Message Passing Iteration', fontsize=14)

        plt.ylabel(y_label, fontsize=15)
        # plt.yscale('symlog')
        plt.yscale('log')
        plt.title(title, fontsize=20)
        # plt.legend(fontsize=12)    
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
                fancybox=True, ncol=3, fontsize=12, prop={'size': 12})
        #make the font bigger
        matplotlib.rcParams.update({'font.size': 10})        

        plt.grid(True)

        plt.savefig('./%s_%s_%d.eps' % (title, y_label, run_grouping), bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')  
        plt.clf()
        
with open('./BPNN_convergence_info.txt', 'r') as load_file:
    (norm_per_isingmodel_vTOf_perIterList_BPNN, norm_per_isingmodel_fTOv_perIterList_BPNN,\
     max_per_isingmodel_vTOf_perIterList_BPNN, max_per_isingmodel_fTOv_perIterList_BPNN) = json.load(load_file)

with open('./BP_convergence_info.txt', 'r') as load_file:
    (norm_per_isingmodel_vTOf_perIterList_BP, norm_per_isingmodel_fTOv_perIterList_BP,\
     max_per_isingmodel_vTOf_perIterList_BP, max_per_isingmodel_fTOv_perIterList_BP) = json.load(load_file)


make_plot(BPNN_data=max_per_isingmodel_fTOv_perIterList_BPNN, BP_data=max_per_isingmodel_fTOv_perIterList_BP)