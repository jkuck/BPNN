import numpy as np

# np.set_printoptions(precision=.0001,suppress=True)

bpnn_errors = [    0.70482,     0.65700,     0.68139,     0.51059,     0.00020,
            0.90185,     0.68924,     0.76599,     0.50119,     1.27491,
            0.63465,     2.07841,     1.91602,     2.03505,     0.74186,
            1.64936,     1.40393,     0.72980,     0.42165,     0.69837,
            0.53225,     0.52792,     0.87350,     0.79021,     0.39832,
            1.38531,     0.83477,     0.06331,     1.12980,     1.19076,
            1.06907,     0.87872,     0.00098,     0.80270,     1.13551,
            0.81858,     0.00027,     0.80485,     0.35765,     0.27380,
            1.28222,     0.61106,     0.82531,     0.04129,     0.79116,
            0.69943,     0.77749,     0.48800,     0.92000,     0.98079]

bp_errors = [    0.21930,     0.79116,     0.06328,   173.76388,     0.52780,
            0.54497,    54.70582,     0.28180,     0.39956,     0.53222,
            0.43547,     0.66821,     0.51540,     0.81864,     0.00095,
            1.72773,     2.07842,     0.35776,     0.42172,    18.07096,
            0.00014,    16.44744,    23.08285,     0.22354,    10.26620,
            0.04136,    20.94101,    37.97716,     0.51065,     0.39100,
            4.10262,     0.38743,     0.35989,     6.50370,     0.39832,
           13.65834,     0.00045,   144.15493,     0.59421,     0.27386,
            9.99670,   532.03391,     0.64789,     0.83479,     0.69182,
            1.19073,   185.24919,    15.74989,     0.63465,     0.48802]

bp_max_per_isingmodel_fTOv = [    0.00000,     0.00000,     0.00000,    24.96452,     0.00001,
            0.00000,    13.77483,     0.00001,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00001,
            0.00000,     0.00001,     0.00000,     0.00001,     0.00000,
            0.00000,     0.00000,     9.89212,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,    19.53122,     0.00001,     0.00000,
            0.00001,    28.85236,     0.00000,     0.00000,     0.00000,
            0.00000,    20.37172,     0.00001,     0.00000,     0.00000]          

bp_max_per_isingmodel_vTOf = [    0.00000,     0.00000,     0.00000,     9.09366,     0.00000,
            0.00000,     4.29004,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,     3.80111,     0.00000,     0.00000,
            0.00000,     0.00000,     0.00000,     0.00000,     0.00000,
            0.00000,     0.00000,     6.80370,     0.00000,     0.00000,
            0.00000,     8.75526,     0.00000,     0.00000,     0.00000,
            0.00000,     7.44560,     0.00000,     0.00000,     0.00000]            

bp_errors_bpConverged = []
bpnn_errors_bpConverged = []
bpErrorMinusBpnnError_bpConverged = []

for idx, bpnn_error in enumerate(bpnn_errors):
	print("Hi")
	if bp_max_per_isingmodel_fTOv[idx] < .0001:
		bp_errors_bpConverged.append(bp_errors[idx])
		bpnn_errors_bpConverged.append(bpnn_errors[idx])
		bpErrorMinusBpnnError_bpConverged.append(bp_errors[idx] - bpnn_errors[idx])

print("bp_errors_bpConverged:", bp_errors_bpConverged)
print("bpnn_errors_bpConverged:", bpnn_errors_bpConverged)
print("bpErrorMinusBpnnError_bpConverged:", bpErrorMinusBpnnError_bpConverged)

print("np.median(bpErrorMinusBpnnError_bpConverged):", np.median(bpErrorMinusBpnnError_bpConverged))
print("sorted(bpErrorMinusBpnnError_bpConverged):", sorted(bpErrorMinusBpnnError_bpConverged))
print("len(bp_errors_bpConverged):", len(bp_errors_bpConverged))


bpnn_errors_bpConverged = np.array(bpnn_errors_bpConverged)
bpnn_squared_errors_bpConverged = np.square(bpnn_errors_bpConverged)
bp_errors_bpConverged = np.array(bp_errors_bpConverged)
bp_squared_errors_bpConverged = np.square(bp_errors_bpConverged)

print("bpnn_squared_errors_bpConverged:", bpnn_squared_errors_bpConverged)
print("bp_squared_errors_bpConverged:", bp_squared_errors_bpConverged)
print("RMSE BPNN, BP converged =", np.sqrt(np.mean(bpnn_squared_errors_bpConverged)))
print("RMSE BP, BP converged =", np.sqrt(np.mean(bp_squared_errors_bpConverged)))





