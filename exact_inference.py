import mrftools
import numpy as np

def calculate_partition_function_exact():
	'''
	Calculate the exact partition function for reference/debugging
	'''
    # clauses = [[1, -2], [-2, 3], [1, 3]] # count=4
    # clauses = [[1, -2], [-1, 2], [-1]] # count=4

    # clauses = [[1, -2, 3], [-2, 4], [3]] # count=6
    # clauses = [[1, -2], [-2, 4], [3]] # count=5
    # clauses = [[1, 2], [-2, 3]] # count=4
    clauses = [[1, 2], [1,-2], [-1,2], [-1,-2]] # count=3

    sat_FactorGraph = mrftools.MarkovNet()

    variables = {}
    for clause in clauses:
        for literal in clause:
            variables[np.abs(literal)] = 1

    for variable, _ in variables.items():
        factor = np.log(np.array([1, 1]))
        sat_FactorGraph.set_unary_factor(variable, factor)

    for clause in clauses:
        if len(clause) == 1:
            literal = clause[0]
            variable = np.abs(literal)
            if literal > 0:
                factor = np.log(np.array([0, 1]))
            else:
                factor = np.log(np.array([1, 0]))
            sat_FactorGraph.set_unary_factor(variable, factor)
        else:
            vars_in_clause = [np.abs(literal) for literal in clause]
            clause_state = np.log(build_clause_node_state(clause=clause, state_dimensions=len(clause)).numpy())
            print("vars_in_clause:", vars_in_clause)
            print("clause_state.shape:", clause_state.shape)
            print("clause_state:", clause_state)
            sat_FactorGraph.set_edge_factor(tuple(vars_in_clause), clause_state)

    sat_FactorGraph.create_matrices()

    bf = mrftools.BruteForce(sat_FactorGraph)

    #FOUND ISSUE: we can only set one factor over a single set of variables, so we get
    #the wrong number of satisfying solutions when multiple clauses contain the exact
    #same variables
    EXPLICIT_BRUTE_FORCE = True
    if EXPLICIT_BRUTE_FORCE:
        z = 0.0

        variables = list(sat_FactorGraph.variables)

        num_states = [sat_FactorGraph.num_states[var] for var in variables]

        arg_list = [range(s) for s in num_states]

        for state_list in itertools.product(*arg_list):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = state_list[i]

            print()
            print("state:", states)

            energy = 0.0
            for var in sat_FactorGraph.variables:
                print("var:", var)
                energy += sat_FactorGraph.unary_potentials[var][states[var]]

                for neighbor in sat_FactorGraph.neighbors[var]:
                    if var < neighbor:
                        energy += sat_FactorGraph.get_potential((var, neighbor))[states[var], states[neighbor]]
                        print("neighbor:", neighbor, "joint cur_potential:",  sat_FactorGraph.get_potential((var, neighbor))[states[var], states[neighbor]])
                print("cur_potential:", sat_FactorGraph.get_potential((var, neighbor))[states[var], states[neighbor]])

            print("energy:", energy)
            print("state value:", np.exp(sat_FactorGraph.evaluate_state(states)))
            z += np.exp(sat_FactorGraph.evaluate_state(states))

        print("z =", z)
        sleep(temp1)


    exact_z = bf.compute_z()
    print('Exact partition sum:', exact_z)


    bp = mrftools.BeliefPropagator(sat_FactorGraph)
    bp.infer(display='full')

    print("Bethe energy functional: %f" % bp.compute_energy_functional())

    print("Brute force log partition function: %f" % np.log(bf.compute_z()))

