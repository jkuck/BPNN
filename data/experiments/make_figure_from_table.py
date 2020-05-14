#this code makes the figures that replaced the latex table
# have to manually enter the numbers from the table

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
x_vals = [i for i in range(9)]
perfect = [0 for i in range(9)]

# 10 x 10 Ising Model Data, attractive field
mean_field = [31.6, 66.9, 472.6, 37.2, 70.4, 433.8, 36.9, 65.6, 437.7]
lbp_1k_seq = [5.9, 23.2 , 100.7 , 8.0 , 17.1 , 119.2 , 9.5 , 10.5 , 109.3]
lbp_1k_par = [10.3, 21.5 , 77.2 , 8.5 , 15.0 , 125.1 , 9.2 , 13.3 , 114.9]
lbp_100_par = [10.5, 21.5 , 77.2 , 9.1 , 15.0 , 125.1 , 9.8 , 13.8 , 114.9 ]
lbp_10_par = [17.9, 19.6 , 23.3 , 11.8 , 15.9 , 19.8 , 7.1 , 10.4 , 11.2 ]
GNN = [.4, 9.5,  184.6,  0.6, 65.6,  168.7,  3.6,  8.5,  186.1 ]
BPNN = [.4, 0.4, 0.7, 0.8, 0.8, 0.9, 3.6, 3.7, 4.5]

# 14 x 14 Ising Model Data, attractive field
# mean_field = [67.4, 167.7, 873.0, 68.0, 163.3, 919.6, 59.7, 168.2, 1019.7]
# lbp_1k_seq = [19.4, 43.8, 183.5, 18.8, 40.3, 191.2, 15.8, 43.8, 216.3]
# lbp_1k_par = [16.5, 40.8, 154.5, 18.7, 37.4, 197.5, 13.0, 40.5, 216.9]
# lbp_100_par = [16.7, 41.0, 156.4, 19.4, 39.2, 197.5, 15.1, 40.7, 216.9]
# lbp_10_par = [38.1, 45.9, 57.7, 27.9, 28.6, 33.8, 18.6, 31.8, 28.7]
# GNN = [2.1, 9.8, 296.7, 1.9, 10.0, 304.3, 8.4, 11.0, 344.4]
# BPNN = [0.9, 0.6, 0.7, 1.0, 1.6, 1.3, 7.3, 8.6, 7.4]


plt.plot(x_vals, perfect, '-', label='Zero Error')

plt.plot(x_vals, mean_field, 'x-', label='Mean Field')
plt.plot(x_vals, lbp_1k_seq, '+:', label='LBP, 1k Sequential Iterations')
# plt.plot(x_vals, lbp_1k_par, 'x-', label='LBP, 1k Par')
# plt.plot(x_vals, lbp_100_par, 'x-', label='LBP, 100 Par')
plt.plot(x_vals, lbp_10_par, '1-.', label='LBP, 10 Parallel Iterations')
plt.plot(x_vals, GNN, '2--', label='GNN')
plt.plot(x_vals, BPNN, 'x-', label='BPNN')

# plt.xlabel('(f_max, c_max)', fontsize=14)
plt.xlabel(r'($f_{max}$, $c_{max}$)', fontsize=14)
# plt.xlabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=14)

plt.ylabel('RMSE', fontsize=15)
plt.yscale('symlog')
plt.title('10x10 Ising Models', fontsize=20)
# plt.legend(fontsize=12)    
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
        fancybox=True, ncol=3, fontsize=12, prop={'size': 12})
#make the font bigger
matplotlib.rcParams.update({'font.size': 10})        

plt.grid(True)

plt.xticks(np.arange(9), ['(.1, 5)', '(.1, 10)', '(.1, 50)', '(.2, 5)', '(.2, 10)', '(.2, 50)', '(1, 5)', '(1, 10)', '(1, 50)'])
# plot_name = 'quick_plot.png'
# plt.savefig(ROOT_DIR + 'sat_plots/' + plot_name) 
plt.savefig('./testplot.eps', bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')   
# plt.show()
