def parse_dimacs(filename, verbose=False):
    clauses = []
    dictionary_of_vars = defaultdict(int)
    # print("parse_dimacs, filename:", filename)  
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split()
            if len(line_as_list) == 0:
                continue
            # print("line_as_list[:-1]:")
            # print(line_as_list[:-1])
            # print()            
            if line_as_list[0] == "p":
                n_vars = int(line_as_list[2])
                n_clauses = int(line_as_list[3])
            elif line_as_list[0] == "c":
                continue
            else:
                cur_clause = [int(s) for s in line_as_list[:-1]]
                for var in cur_clause:
                    dictionary_of_vars[int(abs(var))] += 1
                clauses.append(cur_clause)
    # assert(n_clauses == len(clauses)), (n_clauses, len(clauses), filename)
    if(n_clauses != len(clauses)):
        if verbose:
            print("actual clause count doesn't match expected clause count!!")
    
    #make sure that all variables are named something in [1,...,n_vars]
    for var_name, var_degree in dictionary_of_vars.items():
        assert(var_name <= n_vars and var_name >= 1)

    #create dummy clauses for variables that don't explcitly appear, i.e. if 
    #variable 8 never appears explicitly in a clause this is equivalent to having
    #the additional clause (-8 8 0)
    for var_name in range(1, n_vars+1):
        if var_name not in dictionary_of_vars:
            clauses.append([-var_name, var_name])
            dictionary_of_vars[var_name] = 2
            # print("appended clause:", [-var_name, var_name])
            
# ######DEBUGGING#########
#     N = -1 #number of variables
#     max_var_degree = -1
#     for var_name, var_degree in dictionary_of_vars.items():
#         if var_name > N:
#             N = var_name
#         if var_degree > max_var_degree:
#             max_var_degree = var_degree
#     for var_name, var_degree in dictionary_of_vars.items():
#         if var_name > N:
#             N = var_name
#         if var_degree > max_var_degree:
#             max_var_degree = var_degree
#     if(N != len(dictionary_of_vars)):
#         for var_idx in range(1, N+1):
#             if var_idx not in dictionary_of_vars:
#                 print(var_idx, "missing from dictionary_of_vars")
#     assert(N == len(dictionary_of_vars)), (N, len(dictionary_of_vars), n_vars)



#     dictionary_of_vars_check = defaultdict(int)
#     for clause_idx, clause in enumerate(clauses):
#         for literal in clause:
#             dictionary_of_vars_check[np.abs(literal)] += 1

#     # print("a")

#     # check largest variable name equals the number of variables
#     N = -1 #number of variables
#     max_var_degree = -1
#     for var_name, var_degree in dictionary_of_vars_check.items():
#         if var_name > N:
#             N = var_name
#         if var_degree > max_var_degree:
#             max_var_degree = var_degree
#     if(N != len(dictionary_of_vars_check)):
#         for var_idx in range(1, N+1):
#             if var_idx not in dictionary_of_vars_check:
#                 print(var_idx, "missing from dictionary_of_vars_check")
#     assert(N == len(dictionary_of_vars_check)), (N, len(dictionary_of_vars_check))


# ######END DEBUGGING######

    # assert(len(dictionary_of_vars) == n_vars), (len(dictionary_of_vars), n_vars) #make sure we actually have this many variables

    # if (len(dictionary_of_vars) == n_vars):
    if True: #missing variables imply an always true clause, e.g. (-8 8 0) if 8 is missing
        if verbose:
            print("variable count checks succeeded")
        load_successful = True
    else:
        if verbose:
            print("variable count check failed")
        load_successful = False
        print("load failed for:", filename)
        print("len(dictionary_of_vars):", len(dictionary_of_vars))
        print("n_vars:", n_vars)
        for i in range(1, n_vars+1):
            if i not in dictionary_of_vars:
                print(i, "missing from dictionary_of_vars")
        print()
    return n_vars, clauses, load_successful