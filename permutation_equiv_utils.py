import random

def permute_dim1111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor11 = input_tensor1
    input_tensor111 = input_tensor11
    input_tensor1111 = input_tensor111
    output_tensor1111 = mlp(input_tensor1111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape)
    # assert(output_tensor1111 == input_tensor).all() #for debugging
    return output_tensor1111

def permute_dim1112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor11 = input_tensor1
    input_tensor111 = input_tensor11
    input_tensor1112 = input_tensor111.permute(0,1,2,3,4,6,5)
    output_tensor1112 = mlp(input_tensor1112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5)
    # assert(output_tensor1112 == input_tensor).all() #for debugging
    return output_tensor1112

def permute_dim1121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor11 = input_tensor1
    input_tensor112 = input_tensor11.permute(0,1,2,3,5,4,6) 
    input_tensor1121 = input_tensor112
    output_tensor1121 = mlp(input_tensor1121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6)
    # assert(output_tensor1121 == input_tensor).all() #for debugging
    return output_tensor1121

def permute_dim1122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor11 = input_tensor1
    input_tensor112 = input_tensor11.permute(0,1,2,3,5,4,6) 
    input_tensor1122 = input_tensor112.permute(0,1,2,3,4,6,5)
    output_tensor1122 = mlp(input_tensor1122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6)
    # assert(output_tensor1122 == input_tensor).all() #for debugging
    return output_tensor1122

def permute_dim1131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor11 = input_tensor1
    input_tensor113 = input_tensor11.permute(0,1,2,3,6,5,4)
    input_tensor1131 = input_tensor113
    output_tensor1131 = mlp(input_tensor1131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4)
    # assert(output_tensor1131 == input_tensor).all() #for debugging
    return output_tensor1131

def permute_dim1132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor11 = input_tensor1
    input_tensor113 = input_tensor11.permute(0,1,2,3,6,5,4)
    input_tensor1132 = input_tensor113.permute(0,1,2,3,4,6,5)
    output_tensor1132 = mlp(input_tensor1132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4)
    # assert(output_tensor1132 == input_tensor).all() #for debugging
    return output_tensor1132

def permute_dim1211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor121 = input_tensor12
    input_tensor1211 = input_tensor121
    output_tensor1211 = mlp(input_tensor1211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6)
    # assert(output_tensor1211 == input_tensor).all() #for debugging
    return output_tensor1211

def permute_dim1212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor121 = input_tensor12
    input_tensor1212 = input_tensor121.permute(0,1,2,3,4,6,5)
    output_tensor1212 = mlp(input_tensor1212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6)
    # assert(output_tensor1212 == input_tensor).all() #for debugging
    return output_tensor1212

def permute_dim1221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor122 = input_tensor12.permute(0,1,2,3,5,4,6) 
    input_tensor1221 = input_tensor122
    output_tensor1221 = mlp(input_tensor1221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6)
    # assert(output_tensor1221 == input_tensor).all() #for debugging
    return output_tensor1221

def permute_dim1222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor122 = input_tensor12.permute(0,1,2,3,5,4,6) 
    input_tensor1222 = input_tensor122.permute(0,1,2,3,4,6,5)
    output_tensor1222 = mlp(input_tensor1222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6)
    # assert(output_tensor1222 == input_tensor).all() #for debugging
    return output_tensor1222

def permute_dim1231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor123 = input_tensor12.permute(0,1,2,3,6,5,4)
    input_tensor1231 = input_tensor123
    output_tensor1231 = mlp(input_tensor1231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6)
    # assert(output_tensor1231 == input_tensor).all() #for debugging
    return output_tensor1231

def permute_dim1232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor123 = input_tensor12.permute(0,1,2,3,6,5,4)
    input_tensor1232 = input_tensor123.permute(0,1,2,3,4,6,5)
    output_tensor1232 = mlp(input_tensor1232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6)
    # assert(output_tensor1232 == input_tensor).all() #for debugging
    return output_tensor1232

def permute_dim1311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor131 = input_tensor13
    input_tensor1311 = input_tensor131
    output_tensor1311 = mlp(input_tensor1311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6)
    # assert(output_tensor1311 == input_tensor).all() #for debugging
    return output_tensor1311

def permute_dim1312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor131 = input_tensor13
    input_tensor1312 = input_tensor131.permute(0,1,2,3,4,6,5)
    output_tensor1312 = mlp(input_tensor1312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6)
    # assert(output_tensor1312 == input_tensor).all() #for debugging
    return output_tensor1312

def permute_dim1321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor132 = input_tensor13.permute(0,1,2,3,5,4,6) 
    input_tensor1321 = input_tensor132
    output_tensor1321 = mlp(input_tensor1321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6)
    # assert(output_tensor1321 == input_tensor).all() #for debugging
    return output_tensor1321

def permute_dim1322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor132 = input_tensor13.permute(0,1,2,3,5,4,6) 
    input_tensor1322 = input_tensor132.permute(0,1,2,3,4,6,5)
    output_tensor1322 = mlp(input_tensor1322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6)
    # assert(output_tensor1322 == input_tensor).all() #for debugging
    return output_tensor1322

def permute_dim1331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor133 = input_tensor13.permute(0,1,2,3,6,5,4)
    input_tensor1331 = input_tensor133
    output_tensor1331 = mlp(input_tensor1331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6)
    # assert(output_tensor1331 == input_tensor).all() #for debugging
    return output_tensor1331

def permute_dim1332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor133 = input_tensor13.permute(0,1,2,3,6,5,4)
    input_tensor1332 = input_tensor133.permute(0,1,2,3,4,6,5)
    output_tensor1332 = mlp(input_tensor1332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6)
    # assert(output_tensor1332 == input_tensor).all() #for debugging
    return output_tensor1332

def permute_dim1411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)
    input_tensor141 = input_tensor14
    input_tensor1411 = input_tensor141
    output_tensor1411 = mlp(input_tensor1411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3)
    # assert(output_tensor1411 == input_tensor).all() #for debugging
    return output_tensor1411

def permute_dim1412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)
    input_tensor141 = input_tensor14
    input_tensor1412 = input_tensor141.permute(0,1,2,3,4,6,5)
    output_tensor1412 = mlp(input_tensor1412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3)
    # assert(output_tensor1412 == input_tensor).all() #for debugging
    return output_tensor1412

def permute_dim1421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)
    input_tensor142 = input_tensor14.permute(0,1,2,3,5,4,6) 
    input_tensor1421 = input_tensor142
    output_tensor1421 = mlp(input_tensor1421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3)
    # assert(output_tensor1421 == input_tensor).all() #for debugging
    return output_tensor1421

def permute_dim1422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)
    input_tensor142 = input_tensor14.permute(0,1,2,3,5,4,6) 
    input_tensor1422 = input_tensor142.permute(0,1,2,3,4,6,5)
    output_tensor1422 = mlp(input_tensor1422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3)
    # assert(output_tensor1422 == input_tensor).all() #for debugging
    return output_tensor1422

def permute_dim1431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)
    input_tensor143 = input_tensor14.permute(0,1,2,3,6,5,4)
    input_tensor1431 = input_tensor143
    output_tensor1431 = mlp(input_tensor1431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3)
    # assert(output_tensor1431 == input_tensor).all() #for debugging
    return output_tensor1431

def permute_dim1432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor1 = input_tensor
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)
    input_tensor143 = input_tensor14.permute(0,1,2,3,6,5,4)
    input_tensor1432 = input_tensor143.permute(0,1,2,3,4,6,5)
    output_tensor1432 = mlp(input_tensor1432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3)
    # assert(output_tensor1432 == input_tensor).all() #for debugging
    return output_tensor1432

def permute_dim2111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor21 = input_tensor2
    input_tensor211 = input_tensor21
    input_tensor2111 = input_tensor211
    output_tensor2111 = mlp(input_tensor2111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2111 == input_tensor).all() #for debugging
    return output_tensor2111

def permute_dim2112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor21 = input_tensor2
    input_tensor211 = input_tensor21
    input_tensor2112 = input_tensor211.permute(0,1,2,3,4,6,5)
    output_tensor2112 = mlp(input_tensor2112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2112 == input_tensor).all() #for debugging
    return output_tensor2112

def permute_dim2121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor21 = input_tensor2
    input_tensor212 = input_tensor21.permute(0,1,2,3,5,4,6) 
    input_tensor2121 = input_tensor212
    output_tensor2121 = mlp(input_tensor2121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2121 == input_tensor).all() #for debugging
    return output_tensor2121

def permute_dim2122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor21 = input_tensor2
    input_tensor212 = input_tensor21.permute(0,1,2,3,5,4,6) 
    input_tensor2122 = input_tensor212.permute(0,1,2,3,4,6,5)
    output_tensor2122 = mlp(input_tensor2122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2122 == input_tensor).all() #for debugging
    return output_tensor2122

def permute_dim2131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor21 = input_tensor2
    input_tensor213 = input_tensor21.permute(0,1,2,3,6,5,4)
    input_tensor2131 = input_tensor213
    output_tensor2131 = mlp(input_tensor2131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2131 == input_tensor).all() #for debugging
    return output_tensor2131

def permute_dim2132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor21 = input_tensor2
    input_tensor213 = input_tensor21.permute(0,1,2,3,6,5,4)
    input_tensor2132 = input_tensor213.permute(0,1,2,3,4,6,5)
    output_tensor2132 = mlp(input_tensor2132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2132 == input_tensor).all() #for debugging
    return output_tensor2132

def permute_dim2211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor221 = input_tensor22
    input_tensor2211 = input_tensor221
    output_tensor2211 = mlp(input_tensor2211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2211 == input_tensor).all() #for debugging
    return output_tensor2211

def permute_dim2212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor221 = input_tensor22
    input_tensor2212 = input_tensor221.permute(0,1,2,3,4,6,5)
    output_tensor2212 = mlp(input_tensor2212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2212 == input_tensor).all() #for debugging
    return output_tensor2212

def permute_dim2221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor222 = input_tensor22.permute(0,1,2,3,5,4,6) 
    input_tensor2221 = input_tensor222
    output_tensor2221 = mlp(input_tensor2221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2221 == input_tensor).all() #for debugging
    return output_tensor2221

def permute_dim2222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor222 = input_tensor22.permute(0,1,2,3,5,4,6) 
    input_tensor2222 = input_tensor222.permute(0,1,2,3,4,6,5)
    output_tensor2222 = mlp(input_tensor2222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2222 == input_tensor).all() #for debugging
    return output_tensor2222

def permute_dim2231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor223 = input_tensor22.permute(0,1,2,3,6,5,4)
    input_tensor2231 = input_tensor223
    output_tensor2231 = mlp(input_tensor2231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2231 == input_tensor).all() #for debugging
    return output_tensor2231

def permute_dim2232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor223 = input_tensor22.permute(0,1,2,3,6,5,4)
    input_tensor2232 = input_tensor223.permute(0,1,2,3,4,6,5)
    output_tensor2232 = mlp(input_tensor2232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2232 == input_tensor).all() #for debugging
    return output_tensor2232

def permute_dim2311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor231 = input_tensor23
    input_tensor2311 = input_tensor231
    output_tensor2311 = mlp(input_tensor2311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2311 == input_tensor).all() #for debugging
    return output_tensor2311

def permute_dim2312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor231 = input_tensor23
    input_tensor2312 = input_tensor231.permute(0,1,2,3,4,6,5)
    output_tensor2312 = mlp(input_tensor2312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2312 == input_tensor).all() #for debugging
    return output_tensor2312

def permute_dim2321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor232 = input_tensor23.permute(0,1,2,3,5,4,6) 
    input_tensor2321 = input_tensor232
    output_tensor2321 = mlp(input_tensor2321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2321 == input_tensor).all() #for debugging
    return output_tensor2321

def permute_dim2322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor232 = input_tensor23.permute(0,1,2,3,5,4,6) 
    input_tensor2322 = input_tensor232.permute(0,1,2,3,4,6,5)
    output_tensor2322 = mlp(input_tensor2322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2322 == input_tensor).all() #for debugging
    return output_tensor2322

def permute_dim2331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor233 = input_tensor23.permute(0,1,2,3,6,5,4)
    input_tensor2331 = input_tensor233
    output_tensor2331 = mlp(input_tensor2331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2331 == input_tensor).all() #for debugging
    return output_tensor2331

def permute_dim2332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor233 = input_tensor23.permute(0,1,2,3,6,5,4)
    input_tensor2332 = input_tensor233.permute(0,1,2,3,4,6,5)
    output_tensor2332 = mlp(input_tensor2332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2332 == input_tensor).all() #for debugging
    return output_tensor2332

def permute_dim2411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)
    input_tensor241 = input_tensor24
    input_tensor2411 = input_tensor241
    output_tensor2411 = mlp(input_tensor2411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2411 == input_tensor).all() #for debugging
    return output_tensor2411

def permute_dim2412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)
    input_tensor241 = input_tensor24
    input_tensor2412 = input_tensor241.permute(0,1,2,3,4,6,5)
    output_tensor2412 = mlp(input_tensor2412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2412 == input_tensor).all() #for debugging
    return output_tensor2412

def permute_dim2421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)
    input_tensor242 = input_tensor24.permute(0,1,2,3,5,4,6) 
    input_tensor2421 = input_tensor242
    output_tensor2421 = mlp(input_tensor2421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2421 == input_tensor).all() #for debugging
    return output_tensor2421

def permute_dim2422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)
    input_tensor242 = input_tensor24.permute(0,1,2,3,5,4,6) 
    input_tensor2422 = input_tensor242.permute(0,1,2,3,4,6,5)
    output_tensor2422 = mlp(input_tensor2422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2422 == input_tensor).all() #for debugging
    return output_tensor2422

def permute_dim2431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)
    input_tensor243 = input_tensor24.permute(0,1,2,3,6,5,4)
    input_tensor2431 = input_tensor243
    output_tensor2431 = mlp(input_tensor2431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2431 == input_tensor).all() #for debugging
    return output_tensor2431

def permute_dim2432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)
    input_tensor243 = input_tensor24.permute(0,1,2,3,6,5,4)
    input_tensor2432 = input_tensor243.permute(0,1,2,3,4,6,5)
    output_tensor2432 = mlp(input_tensor2432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    # assert(output_tensor2432 == input_tensor).all() #for debugging
    return output_tensor2432

def permute_dim3111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor31 = input_tensor3
    input_tensor311 = input_tensor31
    input_tensor3111 = input_tensor311
    output_tensor3111 = mlp(input_tensor3111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3111 == input_tensor).all() #for debugging
    return output_tensor3111

def permute_dim3112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor31 = input_tensor3
    input_tensor311 = input_tensor31
    input_tensor3112 = input_tensor311.permute(0,1,2,3,4,6,5)
    output_tensor3112 = mlp(input_tensor3112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3112 == input_tensor).all() #for debugging
    return output_tensor3112

def permute_dim3121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor31 = input_tensor3
    input_tensor312 = input_tensor31.permute(0,1,2,3,5,4,6) 
    input_tensor3121 = input_tensor312
    output_tensor3121 = mlp(input_tensor3121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3121 == input_tensor).all() #for debugging
    return output_tensor3121

def permute_dim3122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor31 = input_tensor3
    input_tensor312 = input_tensor31.permute(0,1,2,3,5,4,6) 
    input_tensor3122 = input_tensor312.permute(0,1,2,3,4,6,5)
    output_tensor3122 = mlp(input_tensor3122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3122 == input_tensor).all() #for debugging
    return output_tensor3122

def permute_dim3131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor31 = input_tensor3
    input_tensor313 = input_tensor31.permute(0,1,2,3,6,5,4)
    input_tensor3131 = input_tensor313
    output_tensor3131 = mlp(input_tensor3131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3131 == input_tensor).all() #for debugging
    return output_tensor3131

def permute_dim3132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor31 = input_tensor3
    input_tensor313 = input_tensor31.permute(0,1,2,3,6,5,4)
    input_tensor3132 = input_tensor313.permute(0,1,2,3,4,6,5)
    output_tensor3132 = mlp(input_tensor3132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3132 == input_tensor).all() #for debugging
    return output_tensor3132

def permute_dim3211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor321 = input_tensor32
    input_tensor3211 = input_tensor321
    output_tensor3211 = mlp(input_tensor3211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3211 == input_tensor).all() #for debugging
    return output_tensor3211

def permute_dim3212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor321 = input_tensor32
    input_tensor3212 = input_tensor321.permute(0,1,2,3,4,6,5)
    output_tensor3212 = mlp(input_tensor3212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3212 == input_tensor).all() #for debugging
    return output_tensor3212

def permute_dim3221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor322 = input_tensor32.permute(0,1,2,3,5,4,6) 
    input_tensor3221 = input_tensor322
    output_tensor3221 = mlp(input_tensor3221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3221 == input_tensor).all() #for debugging
    return output_tensor3221

def permute_dim3222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor322 = input_tensor32.permute(0,1,2,3,5,4,6) 
    input_tensor3222 = input_tensor322.permute(0,1,2,3,4,6,5)
    output_tensor3222 = mlp(input_tensor3222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3222 == input_tensor).all() #for debugging
    return output_tensor3222

def permute_dim3231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor323 = input_tensor32.permute(0,1,2,3,6,5,4)
    input_tensor3231 = input_tensor323
    output_tensor3231 = mlp(input_tensor3231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3231 == input_tensor).all() #for debugging
    return output_tensor3231

def permute_dim3232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor323 = input_tensor32.permute(0,1,2,3,6,5,4)
    input_tensor3232 = input_tensor323.permute(0,1,2,3,4,6,5)
    output_tensor3232 = mlp(input_tensor3232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3232 == input_tensor).all() #for debugging
    return output_tensor3232

def permute_dim3311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor331 = input_tensor33
    input_tensor3311 = input_tensor331
    output_tensor3311 = mlp(input_tensor3311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3311 == input_tensor).all() #for debugging
    return output_tensor3311

def permute_dim3312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor331 = input_tensor33
    input_tensor3312 = input_tensor331.permute(0,1,2,3,4,6,5)
    output_tensor3312 = mlp(input_tensor3312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3312 == input_tensor).all() #for debugging
    return output_tensor3312

def permute_dim3321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor332 = input_tensor33.permute(0,1,2,3,5,4,6) 
    input_tensor3321 = input_tensor332
    output_tensor3321 = mlp(input_tensor3321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3321 == input_tensor).all() #for debugging
    return output_tensor3321

def permute_dim3322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor332 = input_tensor33.permute(0,1,2,3,5,4,6) 
    input_tensor3322 = input_tensor332.permute(0,1,2,3,4,6,5)
    output_tensor3322 = mlp(input_tensor3322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3322 == input_tensor).all() #for debugging
    return output_tensor3322

def permute_dim3331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor333 = input_tensor33.permute(0,1,2,3,6,5,4)
    input_tensor3331 = input_tensor333
    output_tensor3331 = mlp(input_tensor3331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3331 == input_tensor).all() #for debugging
    return output_tensor3331

def permute_dim3332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor333 = input_tensor33.permute(0,1,2,3,6,5,4)
    input_tensor3332 = input_tensor333.permute(0,1,2,3,4,6,5)
    output_tensor3332 = mlp(input_tensor3332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3332 == input_tensor).all() #for debugging
    return output_tensor3332

def permute_dim3411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)
    input_tensor341 = input_tensor34
    input_tensor3411 = input_tensor341
    output_tensor3411 = mlp(input_tensor3411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3411 == input_tensor).all() #for debugging
    return output_tensor3411

def permute_dim3412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)
    input_tensor341 = input_tensor34
    input_tensor3412 = input_tensor341.permute(0,1,2,3,4,6,5)
    output_tensor3412 = mlp(input_tensor3412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3412 == input_tensor).all() #for debugging
    return output_tensor3412

def permute_dim3421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)
    input_tensor342 = input_tensor34.permute(0,1,2,3,5,4,6) 
    input_tensor3421 = input_tensor342
    output_tensor3421 = mlp(input_tensor3421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3421 == input_tensor).all() #for debugging
    return output_tensor3421

def permute_dim3422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)
    input_tensor342 = input_tensor34.permute(0,1,2,3,5,4,6) 
    input_tensor3422 = input_tensor342.permute(0,1,2,3,4,6,5)
    output_tensor3422 = mlp(input_tensor3422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3422 == input_tensor).all() #for debugging
    return output_tensor3422

def permute_dim3431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)
    input_tensor343 = input_tensor34.permute(0,1,2,3,6,5,4)
    input_tensor3431 = input_tensor343
    output_tensor3431 = mlp(input_tensor3431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3431 == input_tensor).all() #for debugging
    return output_tensor3431

def permute_dim3432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)
    input_tensor343 = input_tensor34.permute(0,1,2,3,6,5,4)
    input_tensor3432 = input_tensor343.permute(0,1,2,3,4,6,5)
    output_tensor3432 = mlp(input_tensor3432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    # assert(output_tensor3432 == input_tensor).all() #for debugging
    return output_tensor3432

def permute_dim4111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor41 = input_tensor4
    input_tensor411 = input_tensor41
    input_tensor4111 = input_tensor411
    output_tensor4111 = mlp(input_tensor4111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4111 == input_tensor).all() #for debugging
    return output_tensor4111

def permute_dim4112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor41 = input_tensor4
    input_tensor411 = input_tensor41
    input_tensor4112 = input_tensor411.permute(0,1,2,3,4,6,5)
    output_tensor4112 = mlp(input_tensor4112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4112 == input_tensor).all() #for debugging
    return output_tensor4112

def permute_dim4121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor41 = input_tensor4
    input_tensor412 = input_tensor41.permute(0,1,2,3,5,4,6) 
    input_tensor4121 = input_tensor412
    output_tensor4121 = mlp(input_tensor4121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4121 == input_tensor).all() #for debugging
    return output_tensor4121

def permute_dim4122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor41 = input_tensor4
    input_tensor412 = input_tensor41.permute(0,1,2,3,5,4,6) 
    input_tensor4122 = input_tensor412.permute(0,1,2,3,4,6,5)
    output_tensor4122 = mlp(input_tensor4122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4122 == input_tensor).all() #for debugging
    return output_tensor4122

def permute_dim4131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor41 = input_tensor4
    input_tensor413 = input_tensor41.permute(0,1,2,3,6,5,4)
    input_tensor4131 = input_tensor413
    output_tensor4131 = mlp(input_tensor4131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4131 == input_tensor).all() #for debugging
    return output_tensor4131

def permute_dim4132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor41 = input_tensor4
    input_tensor413 = input_tensor41.permute(0,1,2,3,6,5,4)
    input_tensor4132 = input_tensor413.permute(0,1,2,3,4,6,5)
    output_tensor4132 = mlp(input_tensor4132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4132 == input_tensor).all() #for debugging
    return output_tensor4132

def permute_dim4211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor421 = input_tensor42
    input_tensor4211 = input_tensor421
    output_tensor4211 = mlp(input_tensor4211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4211 == input_tensor).all() #for debugging
    return output_tensor4211

def permute_dim4212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor421 = input_tensor42
    input_tensor4212 = input_tensor421.permute(0,1,2,3,4,6,5)
    output_tensor4212 = mlp(input_tensor4212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4212 == input_tensor).all() #for debugging
    return output_tensor4212

def permute_dim4221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor422 = input_tensor42.permute(0,1,2,3,5,4,6) 
    input_tensor4221 = input_tensor422
    output_tensor4221 = mlp(input_tensor4221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4221 == input_tensor).all() #for debugging
    return output_tensor4221

def permute_dim4222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor422 = input_tensor42.permute(0,1,2,3,5,4,6) 
    input_tensor4222 = input_tensor422.permute(0,1,2,3,4,6,5)
    output_tensor4222 = mlp(input_tensor4222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4222 == input_tensor).all() #for debugging
    return output_tensor4222

def permute_dim4231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor423 = input_tensor42.permute(0,1,2,3,6,5,4)
    input_tensor4231 = input_tensor423
    output_tensor4231 = mlp(input_tensor4231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4231 == input_tensor).all() #for debugging
    return output_tensor4231

def permute_dim4232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor423 = input_tensor42.permute(0,1,2,3,6,5,4)
    input_tensor4232 = input_tensor423.permute(0,1,2,3,4,6,5)
    output_tensor4232 = mlp(input_tensor4232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4232 == input_tensor).all() #for debugging
    return output_tensor4232

def permute_dim4311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor431 = input_tensor43
    input_tensor4311 = input_tensor431
    output_tensor4311 = mlp(input_tensor4311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4311 == input_tensor).all() #for debugging
    return output_tensor4311

def permute_dim4312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor431 = input_tensor43
    input_tensor4312 = input_tensor431.permute(0,1,2,3,4,6,5)
    output_tensor4312 = mlp(input_tensor4312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4312 == input_tensor).all() #for debugging
    return output_tensor4312

def permute_dim4321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor432 = input_tensor43.permute(0,1,2,3,5,4,6) 
    input_tensor4321 = input_tensor432
    output_tensor4321 = mlp(input_tensor4321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4321 == input_tensor).all() #for debugging
    return output_tensor4321

def permute_dim4322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor432 = input_tensor43.permute(0,1,2,3,5,4,6) 
    input_tensor4322 = input_tensor432.permute(0,1,2,3,4,6,5)
    output_tensor4322 = mlp(input_tensor4322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4322 == input_tensor).all() #for debugging
    return output_tensor4322

def permute_dim4331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor433 = input_tensor43.permute(0,1,2,3,6,5,4)
    input_tensor4331 = input_tensor433
    output_tensor4331 = mlp(input_tensor4331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4331 == input_tensor).all() #for debugging
    return output_tensor4331

def permute_dim4332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor433 = input_tensor43.permute(0,1,2,3,6,5,4)
    input_tensor4332 = input_tensor433.permute(0,1,2,3,4,6,5)
    output_tensor4332 = mlp(input_tensor4332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4332 == input_tensor).all() #for debugging
    return output_tensor4332

def permute_dim4411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)
    input_tensor441 = input_tensor44
    input_tensor4411 = input_tensor441
    output_tensor4411 = mlp(input_tensor4411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4411 == input_tensor).all() #for debugging
    return output_tensor4411

def permute_dim4412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)
    input_tensor441 = input_tensor44
    input_tensor4412 = input_tensor441.permute(0,1,2,3,4,6,5)
    output_tensor4412 = mlp(input_tensor4412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4412 == input_tensor).all() #for debugging
    return output_tensor4412

def permute_dim4421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)
    input_tensor442 = input_tensor44.permute(0,1,2,3,5,4,6) 
    input_tensor4421 = input_tensor442
    output_tensor4421 = mlp(input_tensor4421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4421 == input_tensor).all() #for debugging
    return output_tensor4421

def permute_dim4422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)
    input_tensor442 = input_tensor44.permute(0,1,2,3,5,4,6) 
    input_tensor4422 = input_tensor442.permute(0,1,2,3,4,6,5)
    output_tensor4422 = mlp(input_tensor4422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4422 == input_tensor).all() #for debugging
    return output_tensor4422

def permute_dim4431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)
    input_tensor443 = input_tensor44.permute(0,1,2,3,6,5,4)
    input_tensor4431 = input_tensor443
    output_tensor4431 = mlp(input_tensor4431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4431 == input_tensor).all() #for debugging
    return output_tensor4431

def permute_dim4432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)
    input_tensor443 = input_tensor44.permute(0,1,2,3,6,5,4)
    input_tensor4432 = input_tensor443.permute(0,1,2,3,4,6,5)
    output_tensor4432 = mlp(input_tensor4432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    # assert(output_tensor4432 == input_tensor).all() #for debugging
    return output_tensor4432

def permute_dim5111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor51 = input_tensor5
    input_tensor511 = input_tensor51
    input_tensor5111 = input_tensor511
    output_tensor5111 = mlp(input_tensor5111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5111 == input_tensor).all() #for debugging
    return output_tensor5111

def permute_dim5112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor51 = input_tensor5
    input_tensor511 = input_tensor51
    input_tensor5112 = input_tensor511.permute(0,1,2,3,4,6,5)
    output_tensor5112 = mlp(input_tensor5112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5112 == input_tensor).all() #for debugging
    return output_tensor5112

def permute_dim5121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor51 = input_tensor5
    input_tensor512 = input_tensor51.permute(0,1,2,3,5,4,6) 
    input_tensor5121 = input_tensor512
    output_tensor5121 = mlp(input_tensor5121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5121 == input_tensor).all() #for debugging
    return output_tensor5121

def permute_dim5122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor51 = input_tensor5
    input_tensor512 = input_tensor51.permute(0,1,2,3,5,4,6) 
    input_tensor5122 = input_tensor512.permute(0,1,2,3,4,6,5)
    output_tensor5122 = mlp(input_tensor5122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5122 == input_tensor).all() #for debugging
    return output_tensor5122

def permute_dim5131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor51 = input_tensor5
    input_tensor513 = input_tensor51.permute(0,1,2,3,6,5,4)
    input_tensor5131 = input_tensor513
    output_tensor5131 = mlp(input_tensor5131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5131 == input_tensor).all() #for debugging
    return output_tensor5131

def permute_dim5132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor51 = input_tensor5
    input_tensor513 = input_tensor51.permute(0,1,2,3,6,5,4)
    input_tensor5132 = input_tensor513.permute(0,1,2,3,4,6,5)
    output_tensor5132 = mlp(input_tensor5132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5132 == input_tensor).all() #for debugging
    return output_tensor5132

def permute_dim5211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor521 = input_tensor52
    input_tensor5211 = input_tensor521
    output_tensor5211 = mlp(input_tensor5211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5211 == input_tensor).all() #for debugging
    return output_tensor5211

def permute_dim5212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor521 = input_tensor52
    input_tensor5212 = input_tensor521.permute(0,1,2,3,4,6,5)
    output_tensor5212 = mlp(input_tensor5212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5212 == input_tensor).all() #for debugging
    return output_tensor5212

def permute_dim5221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor522 = input_tensor52.permute(0,1,2,3,5,4,6) 
    input_tensor5221 = input_tensor522
    output_tensor5221 = mlp(input_tensor5221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5221 == input_tensor).all() #for debugging
    return output_tensor5221

def permute_dim5222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor522 = input_tensor52.permute(0,1,2,3,5,4,6) 
    input_tensor5222 = input_tensor522.permute(0,1,2,3,4,6,5)
    output_tensor5222 = mlp(input_tensor5222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5222 == input_tensor).all() #for debugging
    return output_tensor5222

def permute_dim5231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor523 = input_tensor52.permute(0,1,2,3,6,5,4)
    input_tensor5231 = input_tensor523
    output_tensor5231 = mlp(input_tensor5231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5231 == input_tensor).all() #for debugging
    return output_tensor5231

def permute_dim5232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor523 = input_tensor52.permute(0,1,2,3,6,5,4)
    input_tensor5232 = input_tensor523.permute(0,1,2,3,4,6,5)
    output_tensor5232 = mlp(input_tensor5232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5232 == input_tensor).all() #for debugging
    return output_tensor5232

def permute_dim5311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor531 = input_tensor53
    input_tensor5311 = input_tensor531
    output_tensor5311 = mlp(input_tensor5311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5311 == input_tensor).all() #for debugging
    return output_tensor5311

def permute_dim5312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor531 = input_tensor53
    input_tensor5312 = input_tensor531.permute(0,1,2,3,4,6,5)
    output_tensor5312 = mlp(input_tensor5312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5312 == input_tensor).all() #for debugging
    return output_tensor5312

def permute_dim5321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor532 = input_tensor53.permute(0,1,2,3,5,4,6) 
    input_tensor5321 = input_tensor532
    output_tensor5321 = mlp(input_tensor5321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5321 == input_tensor).all() #for debugging
    return output_tensor5321

def permute_dim5322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor532 = input_tensor53.permute(0,1,2,3,5,4,6) 
    input_tensor5322 = input_tensor532.permute(0,1,2,3,4,6,5)
    output_tensor5322 = mlp(input_tensor5322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5322 == input_tensor).all() #for debugging
    return output_tensor5322

def permute_dim5331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor533 = input_tensor53.permute(0,1,2,3,6,5,4)
    input_tensor5331 = input_tensor533
    output_tensor5331 = mlp(input_tensor5331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5331 == input_tensor).all() #for debugging
    return output_tensor5331

def permute_dim5332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor533 = input_tensor53.permute(0,1,2,3,6,5,4)
    input_tensor5332 = input_tensor533.permute(0,1,2,3,4,6,5)
    output_tensor5332 = mlp(input_tensor5332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5332 == input_tensor).all() #for debugging
    return output_tensor5332

def permute_dim5411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)
    input_tensor541 = input_tensor54
    input_tensor5411 = input_tensor541
    output_tensor5411 = mlp(input_tensor5411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5411 == input_tensor).all() #for debugging
    return output_tensor5411

def permute_dim5412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)
    input_tensor541 = input_tensor54
    input_tensor5412 = input_tensor541.permute(0,1,2,3,4,6,5)
    output_tensor5412 = mlp(input_tensor5412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5412 == input_tensor).all() #for debugging
    return output_tensor5412

def permute_dim5421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)
    input_tensor542 = input_tensor54.permute(0,1,2,3,5,4,6) 
    input_tensor5421 = input_tensor542
    output_tensor5421 = mlp(input_tensor5421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5421 == input_tensor).all() #for debugging
    return output_tensor5421

def permute_dim5422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)
    input_tensor542 = input_tensor54.permute(0,1,2,3,5,4,6) 
    input_tensor5422 = input_tensor542.permute(0,1,2,3,4,6,5)
    output_tensor5422 = mlp(input_tensor5422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5422 == input_tensor).all() #for debugging
    return output_tensor5422

def permute_dim5431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)
    input_tensor543 = input_tensor54.permute(0,1,2,3,6,5,4)
    input_tensor5431 = input_tensor543
    output_tensor5431 = mlp(input_tensor5431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5431 == input_tensor).all() #for debugging
    return output_tensor5431

def permute_dim5432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)
    input_tensor543 = input_tensor54.permute(0,1,2,3,6,5,4)
    input_tensor5432 = input_tensor543.permute(0,1,2,3,4,6,5)
    output_tensor5432 = mlp(input_tensor5432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    # assert(output_tensor5432 == input_tensor).all() #for debugging
    return output_tensor5432

def var_idx_perm_equivariant_5dfactor_sample(mlp, input_tensor, functions_to_sample=8):
    '''
    A factor containing 5 variables has 5! orderings of the variables.
    Sample functions_to_sample permutations of input_tensor dimensions, apply input mlp to each flattened tensor and unpermute, and return mean to make mlp invariant to variable ordering
    '''
    permutation_functions = [permute_dim1111, permute_dim1112, permute_dim1121, permute_dim1122, permute_dim1131, permute_dim1132, permute_dim1211, permute_dim1212, permute_dim1221, permute_dim1222, permute_dim1231, permute_dim1232, permute_dim1311, permute_dim1312, permute_dim1321, permute_dim1322, permute_dim1331, permute_dim1332, permute_dim1411, permute_dim1412, permute_dim1421, permute_dim1422, permute_dim1431, permute_dim1432, permute_dim2111, permute_dim2112, permute_dim2121, permute_dim2122, permute_dim2131, permute_dim2132, permute_dim2211, permute_dim2212, permute_dim2221, permute_dim2222, permute_dim2231, permute_dim2232, permute_dim2311, permute_dim2312, permute_dim2321, permute_dim2322, permute_dim2331, permute_dim2332, permute_dim2411, permute_dim2412, permute_dim2421, permute_dim2422, permute_dim2431, permute_dim2432, permute_dim3111, permute_dim3112, permute_dim3121, permute_dim3122, permute_dim3131, permute_dim3132, permute_dim3211, permute_dim3212, permute_dim3221, permute_dim3222, permute_dim3231, permute_dim3232, permute_dim3311, permute_dim3312, permute_dim3321, permute_dim3322, permute_dim3331, permute_dim3332, permute_dim3411, permute_dim3412, permute_dim3421, permute_dim3422, permute_dim3431, permute_dim3432, permute_dim4111, permute_dim4112, permute_dim4121, permute_dim4122, permute_dim4131, permute_dim4132, permute_dim4211, permute_dim4212, permute_dim4221, permute_dim4222, permute_dim4231, permute_dim4232, permute_dim4311, permute_dim4312, permute_dim4321, permute_dim4322, permute_dim4331, permute_dim4332, permute_dim4411, permute_dim4412, permute_dim4421, permute_dim4422, permute_dim4431, permute_dim4432, permute_dim5111, permute_dim5112, permute_dim5121, permute_dim5122, permute_dim5131, permute_dim5132, permute_dim5211, permute_dim5212, permute_dim5221, permute_dim5222, permute_dim5231, permute_dim5232, permute_dim5311, permute_dim5312, permute_dim5321, permute_dim5322, permute_dim5331, permute_dim5332, permute_dim5411, permute_dim5412, permute_dim5421, permute_dim5422, permute_dim5431, permute_dim5432]
    sampled_functions = random.sample(permutation_functions, k=functions_to_sample)
    output = sampled_functions[0](mlp, input_tensor)
    for sampled_function in sampled_functions[1:]:
        output += sampled_function(mlp, input_tensor)
    return output/functions_to_sample






