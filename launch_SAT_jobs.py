import subprocess

max_factor_state_dimensions=5
random_seed=1
EXPERIMENT_REPEATS = 1 #run each experiment this time to check variance in results

def launch_set_of_experiments():
#     batch_size = 20
    # for batch_size in [1, 5]:
    #     for train_val_split in ["random_shuffle", "easyTrain_hardVal"]:
    #     #     for problem_category_train in ["or_50_problems"]:
    #         for problem_category_train in ['blasted_problems', 'or_50_problems','problems_75','problems_90','or_60_problems','or_70_problems',\
    #                                        'or_100_problems', 's_problems']:
    #             for msg_passing_iters in [2, 5]:
    # #             for msg_passing_iters in [2, 5, 10]:
    #                 for belief_repeats in [1]:
    # #                 for belief_repeats in [1, 2, 4]:        
    #                     for experiment_repeat_idx in range(EXPERIMENT_REPEATS):
    #                         subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size])

    # problem_category_train = 'blasted_problems'
    # train_val_split = 'random_shuffle'
    for train_val_split in ["random_shuffle", "easyTrain_hardVal"]:
        for batch_size in [5]:
            for lne_mlp in ["True", "False"]:
                for alpha_damping_FtoV in [".5"]:
                    for alpha_damping_VtoF in ["1.0"]:  
                        for belief_repeats in [1, 4]:#, 4]:        
                            for use_MLP1 in [True, False]:
                                for bethe_mlp in ['linear','none']:
                                    for msg_passing_iters in [2, 5]:
                                        for share_weights in [True, False]:
                                            # for problem_category_train in ['blasted_problems', 'or_50_problems','problems_75','problems_90','or_60_problems','or_70_problems',\
                                            #     'or_100_problems', 's_problems']:
                                            for problem_category_train in ['blasted_problems','problems_90','or_100_problems', 's_problems']:                                        
        #                                         for experiment_repeat_idx in range(EXPERIMENT_REPEATS):
                                                use_MLP3 = (not use_MLP1)
                                                subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size, lne_mlp, alpha_damping_FtoV, alpha_damping_VtoF, "%s" % use_MLP1, "%s" % use_MLP3, bethe_mlp, "%s" % share_weights])
            

def launch_batch_size_experiments():
    max_factor_state_dimensions = 3
    msg_passing_iters = 2
    belief_repeats = 1
    random_seed = 1
    problem_category_train = 'blasted_problems'
    train_val_split = 'random_shuffle'
    for batch_size in [1,2,5,10,20,50]:
        subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size])
        
        
def launch_experiments_on_blasted():
    


    problem_category_train = 'blasted_problems'
    train_val_split = 'random_shuffle'
#     for train_val_split in ["random_shuffle", "easyTrain_hardVal"]:
    for batch_size in [1, 5]:
        for lne_mlp in ["True", "False"]:
            for alpha_damping_FtoV in ["1.0", ".5"]:
                for alpha_damping_VtoF in ["1.0"]:  
                    for belief_repeats in [1]:#, 4]:        
                        for use_MLP1 in [True, False]:
                            for bethe_mlp in ['shifted','linear','none']:
                                for msg_passing_iters in [2, 5]:
#                                         for experiment_repeat_idx in range(EXPERIMENT_REPEATS):
                                    use_MLP3 = (not use_MLP1)
                                    subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", "%d" % max_factor_state_dimensions, "%d" % msg_passing_iters, "%d" % belief_repeats, "%d" % random_seed, problem_category_train, train_val_split, "%d" % batch_size, lne_mlp, alpha_damping_FtoV, alpha_damping_VtoF, "%s" % use_MLP1, "%s" % use_MLP3, bethe_mlp])
        

def launch_1experiment_per_category():
    for problem_category_train in ['blasted_problems', 'or_50_problems','problems_75','problems_90','or_60_problems','or_70_problems',\
                                    'or_100_problems', 's_problems']:
        subprocess.run(["sbatch", "launch_bpSAT_sbatch.sh", problem_category_train])
               
# launch_batch_size_experiments()        
launch_set_of_experiments()
# launch_experiments_on_blasted()

# launch_1experiment_per_category()