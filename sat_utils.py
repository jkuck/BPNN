from collections import defaultdict

def parse_dimacs(filename):
    clauses = []
    dictionary_of_vars = defaultdict(int)    
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split(" ")
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
        print("actual clause count doesn't match expected clause count!!")
    #check variables are numbered 1 to n_vars with no gaps        
    num_vars_check = -1
    for var_name, var_degree in dictionary_of_vars.items():
        if var_name > num_vars_check:
            num_vars_check = var_name
    assert(num_vars_check == n_vars) #make sure largest variable is named n_vars
    assert(len(dictionary_of_vars) == n_vars) #make sure we actually have this many variables
    return n_vars, clauses
