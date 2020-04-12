import subprocess

max_factor_state_dimensions=5
random_seed=10
EXPERIMENT_REPEATS = 1 #run each experiment this time to check variance in results

def launch_set_of_experiments():
#     batch_size = 20
    for batch_size in [1, 5]:
        for train_val_split in ["random_shuffle", "easyTrain_hardVal"]:
        #     for problem_category_train in ["or_50_problems"]:
            for problem_category_train in ['blasted_problems', 'or_50_problems','problems_75','problems_90','or_60_problems','or_70_problems',\
                                           'or_100_problems', 's_problems']:
                for msg_passing_iters in [2, 5]:
    #             for msg_passing_iters in [2, 5, 10]:
                    for belief_repeats in [1]:
    #                 for belief_repeats in [1, 2, 4]:        
                        for experiment_repeat_idx in range(EXPERIMENT_REPEATS):
                            subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size])

def launch_batch_size_experiments():
    max_factor_state_dimensions = 3
    msg_passing_iters = 2
    belief_repeats = 1
    random_seed = 1
    problem_category_train = 'blasted_problems'
    train_val_split = 'random_shuffle'
    for batch_size in [1,2,5,10,20,50]:
        subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size])
        
        
# launch_batch_size_experiments()        
launch_set_of_experiments()