import numpy as np
import pickle

results_file = './trained_attrField_10layer_2MLPs_noFinalBetheMLP/ising_model_OOD.pkl'
with open(results_file, 'rb') as f:
    all_results = pickle.load(f)
    
    
print("all_results:")
print(all_results)
for attractive_field in [True, False]:
    for n in [10, 14]:
        for f_max in [.1, .2, 1.0]:
            for c_max in [5.0, 10.0, 50.0]:
                (BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf) = all_results[(attractive_field, n, f_max, c_max)]
                best_col = np.argmin((BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                if best_col == 0:
                    print("%s & %d & %.1f & %.1f & \\textbf{%.1f} & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                elif best_col == 1:
                    print("%s & %d & %.1f & %.1f & %.1f & \\textbf{%.1f} & %.1f & %.1f & %.1f & %.1f & %.1f\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                elif best_col == 2:
                    print("%s & %d & %.1f & %.1f & %.1f & %.1f & \\textbf{%.1f} & %.1f & %.1f & %.1f & %.1f\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                elif best_col == 3:
                    print("%s & %d & %.1f & %.1f & %.1f & %.1f & %.1f & \\textbf{%.1f} & %.1f & %.1f & %.1f\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                elif best_col == 4:
                    print("%s & %d & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & \\textbf{%.1f} & %.1f & %.1f\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                elif best_col == 5:
                    print("%s & %d & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & \\textbf{%.1f} & %.1f\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
                elif best_col == 6:
                    print("%s & %d & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & \\textbf{%.1f}\\\\" % (attractive_field, n, f_max, c_max, BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf))
    print("\\midrule")
